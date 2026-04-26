
Copy

"""
=============================================================
  Siena Scalp EEG — Preprocessing Pipeline v6
  03_preprocess.py
 
  Best configuration from all experiments:
    Channels  : F7, T3, T5, C3, F8, T4, T6, C4  (best ML)
    Window    : 30 seconds
    Step      : 5 seconds (independent windows)
    Preictal  : 5 minutes (most consistent cross-patient)
    + Notch 50Hz
    + Common Average Reference
    + Artifact rejection >500uV
    + Postictal 30min discard
 
  Output: data/processed_v6/
=============================================================
"""
 

import re
import mne
import numpy as np
from pathlib import Path
 
# Suppress MNE's verbose output 
mne.set_log_level("WARNING")
 
 
# File paths
DATA_ROOT  = Path("data/siena-scalp-eeg-database-1.0.0")
 
# save processed windows per patient
OUTPUT_DIR = Path("data/processed_v6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  
 
 
# Configuration constants 
# The 8 electrodes that proved best in all our experiments
# F7, F8 : frontal-temporal — closest to seizure onset zones
# T3, T4 : temporal — core seizure activity
# T5, T6 : posterior temporal — seizure spread
# C3, C4 : central — hemispheric spread detection
TARGET_CHANNELS    = ["F7", "T3", "T5", "C3", "F8", "T4", "T6", "C4"]
 
# Resample everything to 250Hz to match OpenBCI Cyton hardware
# Hospital recorded at 512Hz — we downsample for consistency
TARGET_SFREQ       = 250
 
# Each window is 30 seconds long
# 30s × 250Hz = 7,500 samples per channel per window
WINDOW_SEC         = 30
 
# How far to slide the window each step
# 5s step = consecutive windows share 25s of overlap (83%)
# This is standard — gives more windows without leakage issues
STEP_SEC           = 5
 
# Preictal zone: 0-5 minutes before seizure onset
# Windows in this zone are labeled 1 (preictal)
PREICTAL_SEC       = 5 * 60   # 300 seconds = 5 minutes
 
# Buffer zone: 5-30 minutes before seizure
# Too ambiguous — discard these windows
# (brain may be transitioning but we can't be sure)
BUFFER_SEC         = 30 * 60  # 1800 seconds = 30 minutes
 
# Postictal zone: 30 minutes AFTER seizure ends
# Brain is recovering — not representative of normal state
# Discard to keep interictal data clean
POSTICTAL_SEC      = 30 * 60
 
# Bandpass filter range: keep only brain-relevant frequencies
# Below 0.5Hz = slow drift from patient movement (not brain)
# Above 40Hz  = muscle noise and other artifacts (not brain)
BANDPASS_LOW       = 0.5
BANDPASS_HIGH      = 40.0
 
# Notch filter: remove powerline interference
# Italy (Siena dataset) uses 50Hz power grid
# NOTE: US deployment on Cyton would use 60Hz instead
NOTCH_FREQ         = 50.0
 
# Artifact threshold: reject windows with huge voltage spikes
# >500 microvolts = electrode popping off or patient moving
# Not real brain signal — throw it away
ARTIFACT_THRESHOLD = 500e-6   # 500 microvolts in volts
 
# Pre calculate samples per window and per step
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SEC)   # 7500 samples
STEP_SAMPLES   = int(TARGET_SFREQ * STEP_SEC)     # 1250 samples
 
 
# TIMESTAMP PARSING HELPERS
# The Siena dataset stores seizure times as text like "14.23.45"
# need to convert to seconds from recording start
def extract_first_timestamp(line):
    """
    Extracts a time string (HH.MM.SS) from a line of text.
    Handles edge cases like spaces between digits.
    Returns the timestamp string or None if not found.
    """
    # Remove spaces between digits (e.g. "14 23 45" -> "142345")
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    # Look for pattern like "14.23.45" or "14:23:45"
    match = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None
 
def hms_to_sec(t):
    """
    Converts "HH.MM.SS" string to total seconds.
    e.g. "01.30.00" -> 5400 seconds
    """
    parts = re.split(r"[.:\s]+", t.strip())
    if len(parts) >= 3:
        return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    return 0.0
 
def elapsed_sec(start, end):
    """
    Calculates seconds elapsed from start to end.
    Handles midnight crossover (e.g. 23:59 -> 00:01)
    by adding 24 hours if end < start.
    """
    d = end - start
    return d + 24*3600 if d < 0 else d
 
 
# SEIZURE FILE PARSER
# Each patient has a text file listing all their seizures
# e.g. "Seizures-list-PN07.txt" contains onset/end times
def parse_seizure_file(txt_path):
    """
    Reads the patient's seizure list text file.
    Extracts: which EDF file, onset time, end time per seizure.
    Returns list of seizure dicts.
 
    Only keeps seizures between 20-600 seconds long
    (filters out annotation errors and status epilepticus)
    """
    patient  = txt_path.parent.name
    seizures = []
    with open(txt_path) as f:
        lines = f.readlines()
    current_file = None
    rec_start = sz_start = sz_end = None
 
    def try_commit():
        """Save current seizure if all fields are populated."""
        nonlocal sz_start, sz_end
        if sz_start is not None and sz_end is not None \
                and rec_start is not None:
            onset  = elapsed_sec(rec_start, sz_start)
            sz_dur = elapsed_sec(sz_start, sz_end)
            # Only keep seizures 20-600 seconds long
            if 20 < sz_dur < 600:
                seizures.append({
                    "patient": patient,
                    "file":    current_file or "?",
                    "onset":   onset,        # seconds from recording start
                    "end":     onset + sz_dur,
                    "dur":     sz_dur,
                })
        sz_start = None
        sz_end   = None
 
    for raw_line in lines:
        line = raw_line.strip()
        if not line: continue
        ll = line.lower()
        # Each new "Seizure N X" header triggers saving the previous one
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
    try_commit()  # save the last seizure
    return seizures
 
 
# CHANNEL FINDER
# Maps our desired channel names to whatever the EDF file
# actually calls them (varies between patients/hospitals)
# e.g. we want "T3" but file has "EEG T3-REF"
def find_channels(raw_ch_names, targets):
    """
    Finds the actual EDF channel names that match our targets.
    Strips prefixes like "EEG " and suffixes like "-REF".
    Returns dict mapping target name -> actual EDF channel name.
    """
    mapping = {}
    for target in targets:
        for ch in raw_ch_names:
            # Remove "EEG " prefix
            clean = re.sub(r"^EEG\s*", "", ch).strip()
            # Remove "-REF" or "-LE" suffix
            clean = re.split(r"[-–]", clean)[0].strip()
            if clean.upper() == target.upper():
                mapping[target] = ch
                break
    return mapping
 
 
# WINDOW LABELER
# For each 30-second window, determines its label:
#   1  = preictal (0-5 min before seizure)
#   0  = interictal (>30 min from any seizure)
#  -1  = discard (buffer zone, ictal, or postictal)
def label_window(window_end_sec, seizures_in_file):
    """
    Labels a window based on its position relative to seizures.
 
    window_end_sec   : timestamp (in seconds) of the window's end
    seizures_in_file : list of seizures in the current EDF file
 
    Returns 1 (preictal), 0 (interictal), or -1 (discard)
    """
    for sz in seizures_in_file:
        onset  = sz["onset"]
        sz_end = sz["end"]
 
        # Window overlaps with seizure itself → discard
        if window_end_sec >= onset and window_end_sec <= sz_end:
            return -1
 
        # Window falls in postictal period → discard
        # Brain is recovering — not normal interictal
        if window_end_sec > sz_end and \
                window_end_sec - sz_end <= POSTICTAL_SEC:
            return -1
 
        # Window is after seizure onset → discard
        # (catches any remaining ictal/postictal edge cases)
        if window_end_sec >= onset:
            return -1
 
        time_to_onset = onset - window_end_sec
 
        # Window ends 0-5 min before seizure → PREICTAL 
        if time_to_onset <= PREICTAL_SEC:
            return 1
 
        # Window ends 5-30 min before seizure → buffer → discard
        if time_to_onset <= BUFFER_SEC:
            return -1
 
    # Window is far from all seizures → INTERICTAL 
    return 0
 
 

# EDF FILE PROCESSOR
# Loads one .edf file, applies all preprocessing,
# cuts into labeled 30-second windows
def process_edf(edf_path, seizures_in_file, channel_map):
    """
    Full preprocessing pipeline for one EDF file:
    1. Load raw EEG
    2. Pick our 8 channels
    3. Apply Common Average Reference (noise removal)
    4. Bandpass filter (0.5-40Hz)
    5. Notch filter (50Hz powerline)
    6. Resample to 250Hz
    7. Slice into 30s windows with 5s step
    8. Label each window (preictal/interictal/discard)
    9. Reject artifact windows (>500uV)
    10. Z-score normalize each window
 
    Returns X (windows array) and y (labels array)
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path),
                                   preload=True, verbose=False)
    except Exception as e:
        print(f"    Failed: {e}")
        return None, None
 
    # Keep only our 8 target channels
    available = [ch for ch in channel_map.values()
                 if ch in raw.ch_names]
    if len(available) < 4:
        return None, None  # not enough channels
 
    raw.pick(available)
 
    # Common Average Reference (CAR):
    # Subtract the mean of all channels from each channel
    # Removes noise that affects all electrodes equally
    # (like electrical interference or patient movement)
    raw.set_eeg_reference("average", projection=False, verbose=False)
 
    # Bandpass filter: keep only 0.5-40Hz brain signals
    # IIR = Infinite Impulse Response — fast to compute
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)
 
    # Notch filter: remove 50Hz powerline hum
    raw.notch_filter(NOTCH_FREQ, method="iir", verbose=False)
 
    # Resample to 250Hz if recorded at different rate
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)
 
    # Get raw numpy array: shape (n_channels, n_samples)
    data, _ = raw[:]
    X_list, y_list = [], []
    n_artifact = 0
    start = 0  # sliding window start position in samples
 
    # Slide window across the entire recording
    while start + WINDOW_SAMPLES <= data.shape[1]:
        end   = start + WINDOW_SAMPLES
        # Convert sample position to seconds for labeling
        label = label_window(end / TARGET_SFREQ, seizures_in_file)
 
        if label != -1:  # keep preictal and interictal windows
            window = data[:, start:end].copy()
 
            # Artifact rejection:
            # If any channel has peak-to-peak amplitude > 500uV
            # the window is corrupted — skip it
            if (window.max(axis=1) -
                    window.min(axis=1)).max() > ARTIFACT_THRESHOLD:
                n_artifact += 1
                start += STEP_SAMPLES
                continue
 
            # Z-score normalize each channel within this window
            # Makes each channel have mean=0, std=1
            # Removes differences in absolute signal amplitude
            # between patients and electrode placements
            mean = window.mean(axis=1, keepdims=True)
            std  = window.std(axis=1,  keepdims=True)
            std[std < 1e-10] = 1e-10  # prevent division by zero
            window = (window - mean) / std
 
            X_list.append(window)
            y_list.append(label)
 
        # Move to next window position
        start += STEP_SAMPLES
 
    if n_artifact > 0:
        print(f"      artifacts rejected: {n_artifact}")
    if not X_list:
        return None, None
 
    # Stack list of windows into array
    # X shape: (n_windows, 8_channels, 7500_samples)
    # y shape: (n_windows,) with values 0 or 1
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.int8))
 
 

# MAIN PROCESSING LOOP
# Iterates over all 14 patient folders
print("\n" + "=" * 62)
print("  PREPROCESSING v6")
print("=" * 62)
print(f"  Channels  : {TARGET_CHANNELS}")
print(f"  Window    : {WINDOW_SEC}s  |  Step: {STEP_SEC}s")
print(f"  Preictal  : 0-{PREICTAL_SEC//60} min before seizure")
print(f"  Buffer    : {PREICTAL_SEC//60}-{BUFFER_SEC//60} min (discarded)")
print(f"  Notch     : {NOTCH_FREQ} Hz")
print(f"  Artifact  : >{ARTIFACT_THRESHOLD*1e6:.0f} uV rejected")
print(f"  Reference : Common Average")
print(f"  Output    : {OUTPUT_DIR}/")
print("=" * 62 + "\n")
 
total_pre = total_inter = patients_done = 0
# Find all patient folders (PN00, PN01, ... PN17)
patient_dirs = sorted([d for d in DATA_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("PN")])
 
for pdir in patient_dirs:
    patient = pdir.name
    print(f"-- {patient} " + "-" * 46)
 
    # Find the seizure list text file for this patient
    txt_files = list(pdir.glob("Seizures-list-*.txt"))
    if not txt_files:
        print("  No seizure list -- skipping\n")
        continue
 
    # Parse all seizures for this patient
    all_seizures = parse_seizure_file(txt_files[0])
    if not all_seizures:
        print("  No valid seizures -- skipping\n")
        continue
 
    # Group seizures by which EDF file they appear in
    # (some patients have multiple EDF files)
    seizures_by_file = {}
    for sz in all_seizures:
        seizures_by_file.setdefault(sz["file"], []).append(sz)
 
    # Find all EDF files for this patient
    edf_files = sorted(pdir.glob("*.edf"))
 
    # Probe first EDF to find channel names
    try:
        probe = mne.io.read_raw_edf(str(edf_files[0]),
                                     preload=False, verbose=False)
        channel_map = find_channels(probe.ch_names, TARGET_CHANNELS)
    except Exception as e:
        print(f"  Could not probe: {e} -- skipping\n")
        continue
 
    print(f"  Channels : {len(channel_map)}/8 -> "
          f"{list(channel_map.keys())}")
 
    # Process each EDF file for this patient
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
 
    # Concatenate all EDF files for this patient
    X_all = np.concatenate(patient_X, axis=0)
    y_all = np.concatenate(patient_y, axis=0)
    n_pre   = int((y_all == 1).sum())
    n_inter = int((y_all == 0).sum())
    total_pre   += n_pre
    total_inter += n_inter
    patients_done += 1
 
    # Save patient data as compressed numpy file
    # Shape: X=(n_windows, 8, 7500), y=(n_windows,)
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
 