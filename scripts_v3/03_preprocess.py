"""
=============================================================
  Siena Scalp EEG — Preprocessing Pipeline v3
  03_preprocess.py

  Changes vs v2:
    + Notch filter 50 Hz (removes powerline from beta band)
    + Postictal exclusion (30 min after seizure end discarded)
    + Amplitude artifact rejection (peak-to-peak > 500 µV)
    + Common Average Reference before filtering

  What this script does:
    1. Loads each patient's EDF file
    2. Selects 8 temporal channels
    3. Applies Common Average Reference
    4. Bandpass filters 0.5–40 Hz + notch 50 Hz
    5. Resamples from 512 Hz → 250 Hz
    6. Cuts signal into 30-second windows with 5-second steps
    7. Rejects artifact windows (amplitude > 500 µV peak-to-peak)
    8. Labels each window:
         1 = preictal   (0–5 min before seizure onset)
         0 = interictal (>30 min from any seizure)
        -1 = discard    (5–30 min before, during, or after seizure
                         + 30 min postictal buffer)
    9. Saves a .npz file per patient

  Output folder: data/processed_v3/
=============================================================
"""

import re
import mne
import numpy as np
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT  = Path("data/siena-scalp-eeg-database-1.0.0")
OUTPUT_DIR = Path("data/processed_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHANNELS = ["T3", "T5", "O1", "Pz", "O2", "T6", "T4"]   # headbandf riendly, no frontal

TARGET_SFREQ       = 250
WINDOW_SEC         = 30
STEP_SEC           = 5
PREICTAL_SEC       = 5 * 60
BUFFER_SEC         = 30 * 60
POSTICTAL_SEC      = 30 * 60   # discard 30 min after seizure end
BANDPASS_LOW       = 0.5
BANDPASS_HIGH      = 40.0
NOTCH_FREQ         = 50.0      # EU powerline — change to 60.0 for US
ARTIFACT_THRESHOLD = 500e-6    # 500 µV peak-to-peak

WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SEC)   # 7500
STEP_SAMPLES   = int(TARGET_SFREQ * STEP_SEC)     # 1250


# Seizure file parser

def extract_first_timestamp(line: str):
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    match = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None

def hms_to_sec(t: str) -> float:
    parts = re.split(r"[.:\s]+", t.strip())
    if len(parts) >= 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0

def elapsed_sec(start: float, end: float) -> float:
    d = end - start
    return d + 24 * 3600 if d < 0 else d

def parse_seizure_file(txt_path: Path) -> list:
    patient  = txt_path.parent.name
    seizures = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    current_file = None
    rec_start = rec_end = sz_start = sz_end = None

    def try_commit():
        nonlocal sz_start, sz_end
        if sz_start is not None and sz_end is not None and rec_start is not None:
            onset  = elapsed_sec(rec_start, sz_start)
            sz_dur = elapsed_sec(sz_start, sz_end)
            if 20 < sz_dur < 600:
                seizures.append({
                    "patient": patient,
                    "file":    current_file or "?",
                    "onset":   onset,
                    "end":     onset + sz_dur,
                    "dur":     sz_dur,
                })
        sz_start = None
        sz_end   = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        ll = line.lower()

        if re.match(r"seizure\s+n\s+\d+", ll):
            try_commit()
        elif "file name" in ll:
            p = re.split(r":\s*", line, maxsplit=1)
            if len(p) > 1:
                current_file = p[1].strip()
        elif "registration start" in ll:
            t = extract_first_timestamp(line)
            if t: rec_start = hms_to_sec(t)
        elif "registration end" in ll:
            t = extract_first_timestamp(line)
            if t: rec_end = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?start\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_start = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?end\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_end = hms_to_sec(t)

    try_commit()
    return seizures


# Channel selection 

def find_channels(raw_ch_names: list, targets: list) -> dict:
    mapping = {}
    for target in targets:
        for ch in raw_ch_names:
            clean = re.sub(r"^EEG\s*", "", ch).strip()
            clean = re.split(r"[-–]", clean)[0].strip()
            if clean.upper() == target.upper():
                mapping[target] = ch
                break
    return mapping


# Window labeling

def label_window(window_end_sec: float, seizures_in_file: list) -> int:
    """
    Label a window based on its end time relative to seizures.

    Returns:
       1 = preictal   — window ends 0–5 min before seizure onset
       0 = interictal — window ends >30 min from all seizure events
      -1 = discard    — buffer zone, during seizure, or postictal
    """
    for sz in seizures_in_file:
        onset  = sz["onset"]
        sz_end = sz["end"]

        # During seizure → discard
        if window_end_sec >= onset and window_end_sec <= sz_end:
            return -1

        # Postictal buffer: 30 min after seizure end → discard
        if window_end_sec > sz_end and window_end_sec - sz_end <= POSTICTAL_SEC:
            return -1

        # Window overlaps with or comes after seizure start → discard
        if window_end_sec >= onset:
            return -1

        time_to_onset = onset - window_end_sec

        # Preictal zone: 0–5 min before onset
        if time_to_onset <= PREICTAL_SEC:
            return 1

        # Preictal buffer: 5–30 min before onset → discard
        if time_to_onset <= BUFFER_SEC:
            return -1

    return 0


# Process one EDF file 

def process_edf(edf_path: Path,
                seizures_in_file: list,
                channel_map: dict) -> tuple:
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    Failed to load {edf_path.name}: {e}")
        return None, None

    available = [ch for ch in channel_map.values() if ch in raw.ch_names]
    if len(available) < 4:
        print(f"    Only {len(available)} channels found in {edf_path.name} — skipping")
        return None, None

    raw.pick(available)

    # Phase 1.4 — Common Average Reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Phase 1.1 — Bandpass + notch filter
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)
    raw.notch_filter(NOTCH_FREQ, method="iir", verbose=False)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    data, _ = raw[:]
    n_channels, n_total = data.shape

    X_list, y_list = [], []
    start = 0
    n_artifact = 0

    while start + WINDOW_SAMPLES <= n_total:
        end          = start + WINDOW_SAMPLES
        window_end_s = end / TARGET_SFREQ

        label = label_window(window_end_s, seizures_in_file)

        if label != -1:
            window = data[:, start:end].copy()

            # Phase 1.3 — Amplitude artifact rejection
            if (window.max(axis=1) - window.min(axis=1)).max() > ARTIFACT_THRESHOLD:
                n_artifact += 1
                start += STEP_SAMPLES
                continue

            # Per-channel z-score normalization
            mean = window.mean(axis=1, keepdims=True)
            std  = window.std(axis=1,  keepdims=True)
            std[std < 1e-10] = 1e-10
            window = (window - mean) / std

            X_list.append(window)
            y_list.append(label)

        start += STEP_SAMPLES

    if n_artifact > 0:
        print(f"      artifact windows rejected: {n_artifact}")

    if not X_list:
        return None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.int8))


# Main 

print("\n" + "=" * 62)
print("  PREPROCESSING PIPELINE v3")
print("=" * 62)
print(f"  Channels    : {TARGET_CHANNELS}")
print(f"  Sample rate : {TARGET_SFREQ} Hz  (resampled from 512 Hz)")
print(f"  Window      : {WINDOW_SEC}s  ({WINDOW_SAMPLES} samples)")
print(f"  Step        : {STEP_SEC}s   ({STEP_SAMPLES} samples)")
print(f"  Preictal    : 0–{PREICTAL_SEC//60} min before seizure onset")
print(f"  Buffer      : {PREICTAL_SEC//60}–{BUFFER_SEC//60} min pre-seizure (discarded)")
print(f"  Postictal   : {POSTICTAL_SEC//60} min post-seizure (discarded)")
print(f"  Notch       : {NOTCH_FREQ} Hz")
print(f"  Artifact    : peak-to-peak > {ARTIFACT_THRESHOLD*1e6:.0f} µV → discard")
print(f"  Reference   : Common Average")
print(f"  Output      : {OUTPUT_DIR}/")
print("=" * 62 + "\n")

total_preictal   = 0
total_interictal = 0
patients_done    = 0

patient_dirs = sorted([d for d in DATA_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("PN")])

for pdir in patient_dirs:
    patient = pdir.name
    print(f"── {patient} " + "─" * 46)

    txt_files = list(pdir.glob("Seizures-list-*.txt"))
    if not txt_files:
        print("  No seizure list found — skipping\n")
        continue

    all_seizures = parse_seizure_file(txt_files[0])
    if not all_seizures:
        print("  No valid seizures parsed — skipping\n")
        continue

    seizures_by_file = {}
    for sz in all_seizures:
        seizures_by_file.setdefault(sz["file"], []).append(sz)

    print(f"  Seizures  : {len(all_seizures)}")

    edf_files = sorted(pdir.glob("*.edf"))

    try:
        probe       = mne.io.read_raw_edf(str(edf_files[0]),
                                           preload=False, verbose=False)
        channel_map = find_channels(probe.ch_names, TARGET_CHANNELS)
    except Exception as e:
        print(f"  Could not probe channels: {e} — skipping\n")
        continue

    print(f"  Channels  : {len(channel_map)}/8 mapped → {list(channel_map.keys())}")

    patient_X, patient_y = [], []

    for edf_path in edf_files:
        sz_here = seizures_by_file.get(edf_path.name, [])
        X, y    = process_edf(edf_path, sz_here, channel_map)

        if X is not None:
            patient_X.append(X)
            patient_y.append(y)
            n_pre   = int((y == 1).sum())
            n_inter = int((y == 0).sum())
            print(f"    {edf_path.name:<28} "
                  f"{len(y):>5} windows  "
                  f"preictal: {n_pre:>3}  "
                  f"interictal: {n_inter:>5}")

    if not patient_X:
        print("  No usable windows — skipping\n")
        continue

    X_all = np.concatenate(patient_X, axis=0)
    y_all = np.concatenate(patient_y, axis=0)

    n_pre   = int((y_all == 1).sum())
    n_inter = int((y_all == 0).sum())
    total_preictal   += n_pre
    total_interictal += n_inter
    patients_done    += 1

    print(f"  {patient} done | shape: {X_all.shape} | "
          f"preictal: {n_pre} | interictal: {n_inter}")

    out_path = OUTPUT_DIR / f"{patient}.npz"
    np.savez_compressed(str(out_path), X=X_all, y=y_all)
    print(f"  Saved → {out_path}\n")


print("=" * 62)
print("  DONE")
print("=" * 62)
print(f"  Patients processed : {patients_done} / {len(patient_dirs)}")
print(f"  Total preictal     : {total_preictal:,} windows")
print(f"  Total interictal   : {total_interictal:,} windows")
if total_preictal > 0:
    print(f"  Imbalance ratio    : {total_interictal/total_preictal:.1f}:1")
print(f"  Output             : {OUTPUT_DIR}/")
print("=" * 62 + "\n")
