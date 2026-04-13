"""
=============================================================
  Siena Scalp EEG — EEG Phase Visualization
  06_visualize_eeg_phases.py

  Shows raw EEG signal across 4 phases for one seizure:
    1. Interictal  — normal brain, far from seizure
    2. Preictal    — 0-5 min before seizure onset
    3. Ictal       — during seizure
    4. Postictal   — immediately after seizure

  Uses PN00 seizure 1 as the example (well documented)
  Saves: eeg_phases.png
=============================================================
"""

import re
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")

# Target channels to display 
CHANNELS  = ["T3", "T4", "F7", "F8"]   # 4 key channels for clarity
SFREQ_OUT = 250                          # display at 250 Hz
CLIP_SEC  = 30                           # seconds per phase panel

# Phase definitions (seconds relative to seizure onset) 
# We'll extract a 30-second window from each phase
PHASES = {
    "interictal":  {"offset": -1800, "color": "#4A90D9",  # 30 min before
                    "label": "Interictal\n(30 min before seizure)"},
    "preictal":    {"offset": -180,  "color": "#F5A623",  # 3 min before
                    "label": "Preictal\n(3 min before seizure)"},
    "ictal":       {"offset": 10,    "color": "#D0021B",  # 10s into seizure
                    "label": "Ictal\n(during seizure)"},
    "postictal":   {"offset": 120,   "color": "#7ED321",  # 2 min after
                    "label": "Postictal\n(2 min after seizure)"},
}


# Helpers
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

def find_channel(ch_names, target):
    for ch in ch_names:
        clean = re.sub(r"^EEG\s*", "", ch).strip()
        clean = re.split(r"[-–]", clean)[0].strip()
        if clean.upper() == target.upper():
            return ch
    return None


# Load PN00 seizure 1
# From earlier analysis:
#   File: PN00-1.edf
#   Recording start: 19:39:33
#   Seizure onset:   19:58:36  → 19*60 + 5.05 min into file
#   Seizure end:     19:59:46  → ~70 seconds duration

edf_path    = DATA_ROOT / "PN00" / "PN00-1.edf"
rec_start_s = hms_to_sec("19.39.33")
sz_onset_s  = hms_to_sec("19.58.36")
sz_end_s    = hms_to_sec("19.59.46")

onset_in_file = elapsed_sec(rec_start_s, sz_onset_s)   # ~1143 sec
end_in_file   = elapsed_sec(rec_start_s, sz_end_s)     # ~1213 sec

print(f"Loading {edf_path.name}...")
print(f"  Seizure onset  : {onset_in_file:.0f}s into file "
      f"({onset_in_file/60:.1f} min)")
print(f"  Seizure end    : {end_in_file:.0f}s into file")
print(f"  Seizure duration: {end_in_file-onset_in_file:.0f}s")

raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

# Resample and filter
raw.resample(SFREQ_OUT, verbose=False)
raw.filter(0.5, 40.0, method="iir", verbose=False)

# Find channels
ch_map = {}
for target in CHANNELS:
    found = find_channel(raw.ch_names, target)
    if found:
        ch_map[target] = found
    else:
        print(f"  Channel {target} not found")

available = list(ch_map.values())
raw.pick(available)
print(f"  Channels: {list(ch_map.keys())}")

data, times = raw[:]
total_dur = data.shape[1] / SFREQ_OUT



# Extract phase clips
clips = {}
for phase_name, phase_info in PHASES.items():
    clip_start = onset_in_file + phase_info["offset"]
    clip_end   = clip_start + CLIP_SEC

    # Clamp to file bounds
    clip_start = max(0, clip_start)
    clip_end   = min(total_dur, clip_end)

    if clip_end <= clip_start:
        print(f"  Phase {phase_name} out of bounds — skipping")
        continue

    s_idx = int(clip_start * SFREQ_OUT)
    e_idx = int(clip_end   * SFREQ_OUT)

    clip_data = data[:, s_idx:e_idx] * 1e6  # convert to microvolts
    clip_time = np.linspace(0, CLIP_SEC, e_idx - s_idx)

    clips[phase_name] = {
        "data":  clip_data,
        "time":  clip_time,
        "info":  phase_info,
        "start": clip_start,
    }
    print(f"  Extracted {phase_name}: {clip_start:.0f}s – {clip_end:.0f}s")


# Plot
n_phases   = len(clips)
n_channels = len(available)

fig, axes = plt.subplots(
    n_channels, n_phases,
    figsize=(20, 10),
    sharey=False
)

fig.suptitle(
    "EEG Signal Across Seizure Phases — Patient PN00, Seizure 1\n"
    "Channels: T3 (left temporal), T4 (right temporal), "
    "F7 (left frontal-temporal), F8 (right frontal-temporal)",
    fontsize=13, fontweight="bold", y=1.01
)

phase_names = list(clips.keys())

for col, phase_name in enumerate(phase_names):
    clip      = clips[phase_name]
    color     = clip["info"]["color"]
    label     = clip["info"]["label"]
    clip_data = clip["data"]
    clip_time = clip["time"]

    for row, (target, actual_ch) in enumerate(ch_map.items()):
        ax = axes[row, col]
        ch_idx = list(raw.ch_names).index(actual_ch)

        # Plot signal
        ax.plot(clip_time, clip_data[ch_idx],
                color=color, linewidth=0.6, alpha=0.9)

        # Shade background
        ax.set_facecolor(color + "18")  # very light tint

        # Labels
        if row == 0:
            ax.set_title(label, fontsize=10,
                         fontweight="bold", color=color, pad=8)
        if col == 0:
            ax.set_ylabel(f"{target}\n(µV)", fontsize=9,
                          rotation=0, labelpad=45, va="center")
        if row == n_channels - 1:
            ax.set_xlabel("Time (s)", fontsize=8)

        # Y axis limits — auto per channel per phase for visibility
        signal    = clip_data[ch_idx]
        amplitude = np.percentile(np.abs(signal), 98)
        ylim      = max(amplitude * 1.3, 50)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")

        # Clean up axes
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, CLIP_SEC)

        # Add amplitude annotation
        ax.text(0.98, 0.95, f"±{amplitude:.0f}µV",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=7, color="gray")

# Add phase separator lines
for col in range(1, n_phases):
    line = plt.Line2D(
        [col/n_phases, col/n_phases], [0, 1],
        transform=fig.transFigure,
        color="lightgray", linewidth=1.5, linestyle="--"
    )
    fig.add_artist(line)

# Legend
patches = [
    mpatches.Patch(color=clips[p]["info"]["color"],
                   label=clips[p]["info"]["label"].replace("\n", " "))
    for p in phase_names
]
fig.legend(handles=patches, loc="lower center",
           ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.04),
           frameon=True)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("eeg_phases.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to eeg_phases.png")
print("\nWhat to look for:")
print("  Interictal : irregular, low-amplitude, complex signal")
print("  Preictal   : subtle changes — slight increase in slow waves")
print("  Ictal      : dramatic high-amplitude rhythmic spikes")
print("  Postictal  : slow, suppressed signal — brain recovering")