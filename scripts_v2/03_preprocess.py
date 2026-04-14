"""
=============================================================
  Siena Scalp EEG — Preprocessing Pipeline
  03_preprocess.py

  What this script does:
    1. Loads each patient's EDF file
    2. Selects 6 temporal channels (F7,T3,T5,F8,T4,T6)
    3. Bandpass filters 0.5–40 Hz
    4. Resamples from 512 Hz → 250 Hz (matches Cyton hardware)
    5. Cuts signal into 30-second windows with 5-second steps
    6. Labels each window:
         1 = preictal   (0–5 min before seizure onset)
         0 = interictal (>30 min from any seizure)
        -1 = discard    (5–30 min before, during, or after seizure)
    7. Saves a .npz file per patient:
         X shape: (n_windows, 6, 7500)
         y shape: (n_windows,)

  Output folder: data/processed/
=============================================================
"""

import re
import mne
import numpy as np
from pathlib import Path

mne.set_log_level("WARNING")

# Config
DATA_ROOT   = Path("data/siena-scalp-eeg-database-1.0.0")
OUTPUT_DIR  = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 channel bilateral temporal montage
TARGET_CHANNELS = ["F7", "T3", "T5", "C3", "F8", "T4", "T6", "C4"]

TARGET_SFREQ  = 250      # Hz matches Cyton board
WINDOW_SEC    = 30       # seconds per window
STEP_SEC      = 5        # step between windows
PREICTAL_SEC  = 5 * 60   # 5 min before seizure onset = preictal label
BUFFER_SEC    = 30 * 60  # 5–30 min before seizure = discard zone
BANDPASS_LOW  = 0.5      # Hz
BANDPASS_HIGH = 40.0     # Hz

WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SEC)  # 7500
STEP_SAMPLES   = int(TARGET_SFREQ * STEP_SEC)    # 1250


# HELPERS — seizure file parser (same logic as 02/03 scripts)
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
    """Time elapsed from start to end, handling midnight rollover."""
    d = end - start
    return d + 24 * 3600 if d < 0 else d

def parse_seizure_file(txt_path: Path) -> list:
    """
    Parse a Seizures-list-PNxx.txt file.
    Returns a list of dicts with seizure onset/end in seconds
    relative to the start of each EDF recording file.
    """
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


# Channel select
# Handles name variations e.g. "EEG T3", "T3-A1", "T3"
def find_channels(raw_ch_names: list, targets: list) -> dict:
    """
    Returns dict: target_name → actual channel name in EDF.
    Only includes targets that were actually found.
    """
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
       0 = interictal — window ends >30 min from all seizures
      -1 = discard    — ambiguous, during, or post-seizure
    """
    for sz in seizures_in_file:
        onset  = sz["onset"]
        sz_end = sz["end"]

        # Window overlaps with or comes after seizure start → discard
        if window_end_sec >= onset:
            return -1

        time_to_onset = onset - window_end_sec

        # Within preictal zone (0–5 min before onset) → preictal
        if time_to_onset <= PREICTAL_SEC:
            return 1

        # Within buffer zone (5–30 min before onset) → discard
        if time_to_onset <= BUFFER_SEC:
            return -1

    # More than 30 min from all seizures → interictal
    return 0


# Process one EDF file
def process_edf(edf_path: Path,
                seizures_in_file: list,
                channel_map: dict) -> tuple:
    """
    Load → filter → resample → window → label one EDF file.
    Returns (X, y) numpy arrays or (None, None) if unusable.

    X shape: (n_windows, n_channels, WINDOW_SAMPLES)
    y shape: (n_windows,)
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"Failed to load {edf_path.name}: {e}")
        return None, None

    # Pick only our target channels
    available = [ch for ch in channel_map.values() if ch in raw.ch_names]
    if len(available) < 4:
        print(f"Only {len(available)} target channels found "
              f"in {edf_path.name} — skipping")
        return None, None

    raw.pick(available)

    # Bandpass filter 0.5–40 Hz
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)

    # Resample to 250 Hz (match Cyton)
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    # Extract numpy array (channels × samples) 
    data, _ = raw[:]
    n_channels, n_total = data.shape

    # Sliding window 
    X_list, y_list = [], []
    start = 0

    while start + WINDOW_SAMPLES <= n_total:
        end          = start + WINDOW_SAMPLES
        window_end_s = end / TARGET_SFREQ

        label = label_window(window_end_s, seizures_in_file)

        if label != -1:
            window = data[:, start:end].copy()  # (channels, 7500)

            # Z-score normalize per channel
            mean = window.mean(axis=1, keepdims=True)
            std  = window.std(axis=1,  keepdims=True)
            std[std < 1e-10] = 1e-10
            window = (window - mean) / std

            X_list.append(window)
            y_list.append(label)

        start += STEP_SAMPLES

    if not X_list:
        return None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.int8))


# Main loop
print("\n" + "=" * 62)
print("  PREPROCESSING PIPELINE")
print("=" * 62)
print(f"  Channels    : {TARGET_CHANNELS}")
print(f"  Sample rate : {TARGET_SFREQ} Hz  (resampled from 512 Hz)")
print(f"  Window      : {WINDOW_SEC}s  ({WINDOW_SAMPLES} samples)")
print(f"  Step        : {STEP_SEC}s   ({STEP_SAMPLES} samples)")
print(f"  Preictal    : 0–{PREICTAL_SEC//60} min before seizure onset")
print(f"  Buffer      : {PREICTAL_SEC//60}–{BUFFER_SEC//60} min (discarded)")
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

    # Parse seizure labels
    txt_files = list(pdir.glob("Seizures-list-*.txt"))
    if not txt_files:
        print("No seizure list found — skipping\n")
        continue

    all_seizures = parse_seizure_file(txt_files[0])
    if not all_seizures:
        print("No valid seizures parsed — skipping\n")
        continue

    # Group seizures by EDF filename
    seizures_by_file = {}
    for sz in all_seizures:
        seizures_by_file.setdefault(sz["file"], []).append(sz)

    print(f"  Seizures  : {len(all_seizures)}")

    # Find EDF files 
    edf_files = sorted(pdir.glob("*.edf"))

    # Map channel names using first EDF as probe 
    try:
        probe       = mne.io.read_raw_edf(str(edf_files[0]),
                                           preload=False, verbose=False)
        channel_map = find_channels(probe.ch_names, TARGET_CHANNELS)
    except Exception as e:
        print(f"Could not probe channels: {e} — skipping\n")
        continue

    print(f"  Channels  : {len(channel_map)}/7 mapped → {list(channel_map.keys())}")

    # Process each EDF
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
        print("No usable windows — skipping\n")
        continue

    # Concatenate all files for this patient 
    X_all = np.concatenate(patient_X, axis=0)
    y_all = np.concatenate(patient_y, axis=0)

    n_pre   = int((y_all == 1).sum())
    n_inter = int((y_all == 0).sum())
    total_preictal   += n_pre
    total_interictal += n_inter
    patients_done    += 1

    print(f"{patient} done | "
          f"shape: {X_all.shape} | "
          f"preictal: {n_pre} | "
          f"interictal: {n_inter}")

    # Save compressed 
    out_path = OUTPUT_DIR / f"{patient}.npz"
    np.savez_compressed(str(out_path), X=X_all, y=y_all)
    print(f"Saved - {out_path}\n")


# Summary
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