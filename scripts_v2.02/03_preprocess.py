"""
=============================================================
  Siena Scalp EEG — Preprocessing Pipeline v2.02 FINAL
  03_preprocess.py

  Optimal configuration based on all experiments:
    Channels  : F7, T3, T5, C3, F8, T4, T6, C4
    Window    : 30 seconds
    Step      : 5 seconds (NOT 2s — causes 93% overlap)
    Preictal  : 8 minutes (was 5 — 60% more preictal data)
    Buffer    : 30 minutes
    Postictal : 30 minutes discarded
    + Notch 50Hz
    + Common Average Reference
    + Artifact rejection >500uV
=============================================================
"""

import re
import mne
import numpy as np
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT  = Path("data/siena-scalp-eeg-database-1.0.0")
OUTPUT_DIR = Path("data/processed_v202")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHANNELS    = ["F7", "T3", "T5", "C3", "F8", "T4", "T6", "C4"]
TARGET_SFREQ       = 250
WINDOW_SEC         = 30
STEP_SEC           = 5        # 5s step = independent windows
PREICTAL_SEC       = 8 * 60   # 8 min = ~96 windows per seizure
BUFFER_SEC         = 30 * 60
POSTICTAL_SEC      = 30 * 60
BANDPASS_LOW       = 0.5
BANDPASS_HIGH      = 40.0
NOTCH_FREQ         = 50.0
ARTIFACT_THRESHOLD = 500e-6

WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SEC)   # 7500
STEP_SAMPLES   = int(TARGET_SFREQ * STEP_SEC)     # 1250


def extract_first_timestamp(line):
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    match = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None

def hms_to_sec(t):
    parts = re.split(r"[.:\s]+", t.strip())
    if len(parts) >= 3:
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    return 0.0

def elapsed_sec(start, end):
    d = end - start
    return d + 24*3600 if d < 0 else d

def parse_seizure_file(txt_path):
    patient  = txt_path.parent.name
    seizures = []
    with open(txt_path) as f:
        lines = f.readlines()
    current_file = None
    rec_start = sz_start = sz_end = None

    def try_commit():
        nonlocal sz_start, sz_end
        if sz_start is not None and sz_end is not None \
                and rec_start is not None:
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
        if not line: continue
        ll = line.lower()
        if re.match(r"seizure\s+n\s+\d+", ll):
            try_commit()
        elif "file name" in ll:
            p = re.split(r":\s*", line, maxsplit=1)
            if len(p) > 1: current_file = p[1].strip()
        elif "registration start" in ll:
            t = extract_first_timestamp(line)
            if t: rec_start = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?start\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_start = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?end\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_end = hms_to_sec(t)
    try_commit()
    return seizures

def find_channels(raw_ch_names, targets):
    mapping = {}
    for target in targets:
        for ch in raw_ch_names:
            clean = re.sub(r"^EEG\s*", "", ch).strip()
            clean = re.split(r"[-–]", clean)[0].strip()
            if clean.upper() == target.upper():
                mapping[target] = ch
                break
    return mapping

def label_window(window_end_sec, seizures_in_file):
    for sz in seizures_in_file:
        onset  = sz["onset"]
        sz_end = sz["end"]
        if window_end_sec >= onset and window_end_sec <= sz_end:
            return -1
        if window_end_sec > sz_end and \
                window_end_sec - sz_end <= POSTICTAL_SEC:
            return -1
        if window_end_sec >= onset:
            return -1
        time_to_onset = onset - window_end_sec
        if time_to_onset <= PREICTAL_SEC:
            return 1
        if time_to_onset <= BUFFER_SEC:
            return -1
    return 0

def process_edf(edf_path, seizures_in_file, channel_map):
    try:
        raw = mne.io.read_raw_edf(str(edf_path),
                                   preload=True, verbose=False)
    except Exception as e:
        print(f"    Failed: {e}")
        return None, None

    available = [ch for ch in channel_map.values()
                 if ch in raw.ch_names]
    if len(available) < 4:
        return None, None

    raw.pick(available)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)
    raw.notch_filter(NOTCH_FREQ, method="iir", verbose=False)
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    data, _ = raw[:]
    X_list, y_list = [], []
    n_artifact = 0
    start = 0

    while start + WINDOW_SAMPLES <= data.shape[1]:
        end   = start + WINDOW_SAMPLES
        label = label_window(end / TARGET_SFREQ, seizures_in_file)
        if label != -1:
            window = data[:, start:end].copy()
            if (window.max(axis=1) -
                    window.min(axis=1)).max() > ARTIFACT_THRESHOLD:
                n_artifact += 1
                start += STEP_SAMPLES
                continue
            mean = window.mean(axis=1, keepdims=True)
            std  = window.std(axis=1,  keepdims=True)
            std[std < 1e-10] = 1e-10
            window = (window - mean) / std
            X_list.append(window)
            y_list.append(label)
        start += STEP_SAMPLES

    if n_artifact > 0:
        print(f"      artifacts rejected: {n_artifact}")
    if not X_list:
        return None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int8))


print("\n" + "=" * 62)
print("  PREPROCESSING v2.02 FINAL")
print("=" * 62)
print(f"  Channels  : {TARGET_CHANNELS}")
print(f"  Window    : {WINDOW_SEC}s")
print(f"  Step      : {STEP_SEC}s  (independent windows)")
print(f"  Preictal  : 0-{PREICTAL_SEC//60} min before seizure")
print(f"  Buffer    : {PREICTAL_SEC//60}-{BUFFER_SEC//60} min (discarded)")
print(f"  Notch     : {NOTCH_FREQ} Hz")
print(f"  Artifact  : >{ARTIFACT_THRESHOLD*1e6:.0f} uV rejected")
print(f"  Reference : Common Average")
print("=" * 62 + "\n")

total_pre = total_inter = patients_done = 0
patient_dirs = sorted([d for d in DATA_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("PN")])

for pdir in patient_dirs:
    patient = pdir.name
    print(f"-- {patient} " + "-" * 46)
    txt_files = list(pdir.glob("Seizures-list-*.txt"))
    if not txt_files:
        print("  No seizure list -- skipping\n")
        continue
    all_seizures = parse_seizure_file(txt_files[0])
    if not all_seizures:
        print("  No valid seizures -- skipping\n")
        continue

    seizures_by_file = {}
    for sz in all_seizures:
        seizures_by_file.setdefault(sz["file"], []).append(sz)

    edf_files = sorted(pdir.glob("*.edf"))
    try:
        probe = mne.io.read_raw_edf(str(edf_files[0]),
                                     preload=False, verbose=False)
        channel_map = find_channels(probe.ch_names, TARGET_CHANNELS)
    except Exception as e:
        print(f"  Could not probe: {e} -- skipping\n")
        continue

    print(f"  Channels : {len(channel_map)}/8 -> "
          f"{list(channel_map.keys())}")

    patient_X, patient_y = [], []
    for edf_path in edf_files:
        sz_here = seizures_by_file.get(edf_path.name, [])
        X, y    = process_edf(edf_path, sz_here, channel_map)
        if X is not None:
            patient_X.append(X)
            patient_y.append(y)
            print(f"    {edf_path.name:<28} "
                  f"{len(y):>5} windows  "
                  f"pre:{int((y==1).sum()):>4}  "
                  f"inter:{int((y==0).sum()):>6}")

    if not patient_X:
        print("  No usable windows -- skipping\n")
        continue

    X_all = np.concatenate(patient_X, axis=0)
    y_all = np.concatenate(patient_y, axis=0)
    n_pre   = int((y_all == 1).sum())
    n_inter = int((y_all == 0).sum())
    total_pre   += n_pre
    total_inter += n_inter
    patients_done += 1

    out_path = OUTPUT_DIR / f"{patient}.npz"
    np.savez_compressed(str(out_path), X=X_all, y=y_all)
    print(f"  Saved -> pre:{n_pre}  inter:{n_inter}\n")

print("=" * 62)
print("  DONE")
print("=" * 62)
print(f"  Patients   : {patients_done} / {len(patient_dirs)}")
print(f"  Preictal   : {total_pre:,} windows")
print(f"  Interictal : {total_inter:,} windows")
if total_pre > 0:
    print(f"  Imbalance  : {total_inter/total_pre:.1f}:1")
print(f"  Output     : {OUTPUT_DIR}/")
print("=" * 62 + "\n")