"""
=============================================================
  Siena Scalp EEG — Seizure Close-Up Visualization
  09_seizure_closeup.py

  Best patient: PN07 (AUC 0.884)
  Shows three phases with high time resolution:
    1. Interictal  — 10 min before seizure (normal baseline)
    2. Preictal    — 1 min before seizure onset
    3. Ictal       — during seizure

  Also shows:
    - Band power evolution over time (how features change)
    - Frequency spectrum comparison between phases
=============================================================
"""

import re
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")

# PN07 seizure info from parsed data
# File: PN07-1.edf
# Seizure onset: ~367.65 min into file = 22059 seconds
PATIENT      = "PN07"
EDF_FILE     = "PN07-1.edf"
ONSET_IN_FILE = 367.65 * 60   # seconds into file

SFREQ_OUT    = 250
CHANNELS     = ["T3", "T4", "F7", "F8"]
BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 40.0

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# Phase clip definitions
PHASES = {
    "interictal": {
        "offset": -600,   # 10 min before
        "duration": 30,
        "color": "#4A90D9",
        "label": "Interictal (10 min before)"
    },
    "preictal_1min": {
        "offset": -60,    # 1 min before
        "duration": 30,
        "color": "#F5A623",
        "label": "Preictal (1 min before)"
    },
    "ictal": {
        "offset": 5,      # 5 seconds into seizure
        "duration": 30,
        "color": "#D0021B",
        "label": "Ictal (during seizure)"
    },
}


# Helpers
def find_channel(ch_names, target):
    for ch in ch_names:
        clean = re.sub(r"^EEG\s*", "", ch).strip()
        clean = re.split(r"[-–]", clean)[0].strip()
        if clean.upper() == target.upper():
            return ch
    return None

def compute_band_powers(data_1ch, sfreq):
    """Compute band powers for a single channel signal."""
    freqs, psd = signal.welch(data_1ch, fs=sfreq, nperseg=512)
    powers = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        powers[band] = np.log1p(psd[mask].mean())
    return powers, freqs, psd


# Load data
edf_path = DATA_ROOT / PATIENT / EDF_FILE
print(f"Loading {edf_path.name}...")

raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
raw.resample(SFREQ_OUT, verbose=False)
raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)

# Find channels
ch_map = {}
for target in CHANNELS:
    found = find_channel(raw.ch_names, target)
    if found:
        ch_map[target] = found

print(f"Channels found: {list(ch_map.keys())}")
raw.pick(list(ch_map.values()))
data, times = raw[:]
data_uv = data * 1e6  # microvolts
total_dur = data.shape[1] / SFREQ_OUT

print(f"Seizure onset: {ONSET_IN_FILE/60:.1f} min into file")
print(f"Total duration: {total_dur/60:.1f} min")


# Extract phase clips
clips = {}
for phase_name, phase_info in PHASES.items():
    start = ONSET_IN_FILE + phase_info["offset"]
    end   = start + phase_info["duration"]
    start = max(0, start)
    end   = min(total_dur - 1, end)

    s_idx = int(start * SFREQ_OUT)
    e_idx = int(end   * SFREQ_OUT)

    clips[phase_name] = {
        "data": data_uv[:, s_idx:e_idx],
        "time": np.linspace(0, phase_info["duration"],
                            e_idx - s_idx),
        "info": phase_info,
    }
    print(f"  {phase_name}: {start:.0f}s – {end:.0f}s")


# Figure layout
# Three sections:
#   Top    — raw EEG per phase (4 channels × 3 phases)
#   Middle — frequency spectrum comparison (1 per channel)
#   Bottom — band power bar chart comparison
fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    f"EEG Seizure Close-Up — Patient {PATIENT}  |  "
    f"Best LOPO AUC: 0.884\n"
    f"Seizure onset at {ONSET_IN_FILE/60:.1f} min into recording",
    fontsize=14, fontweight="bold", y=0.98
)

n_ch     = len(ch_map)
n_phases = len(clips)
phase_list = list(clips.keys())
ch_list    = list(ch_map.keys())

# Section 1: Raw EEG 
gs_top = gridspec.GridSpec(
    n_ch, n_phases,
    top=0.88, bottom=0.55,
    hspace=0.15, wspace=0.05
)

for col, phase_name in enumerate(phase_list):
    clip  = clips[phase_name]
    color = clip["info"]["color"]
    label = clip["info"]["label"]

    for row, ch_name in enumerate(ch_list):
        ax = fig.add_subplot(gs_top[row, col])
        ax.plot(clip["time"], clip["data"][row],
                color=color, linewidth=0.6, alpha=0.9)
        ax.set_facecolor(color + "15")

        if row == 0:
            ax.set_title(label, fontsize=10,
                         fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=8,
                          rotation=0, labelpad=40, va="center")
        if row == n_ch - 1:
            ax.set_xlabel("Time (s)", fontsize=8)

        sig = clip["data"][row]
        amp = np.percentile(np.abs(sig), 98)
        ylim = max(amp * 1.4, 30)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.text(0.98, 0.95, f"±{amp:.0f}µV",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray")

# Section 2: Frequency Spectrum 
gs_mid = gridspec.GridSpec(
    1, n_ch,
    top=0.50, bottom=0.30,
    hspace=0.3, wspace=0.3
)

phase_colors = [clips[p]["info"]["color"] for p in phase_list]
phase_labels = [clips[p]["info"]["label"] for p in phase_list]

for col, ch_name in enumerate(ch_list):
    ax = fig.add_subplot(gs_mid[0, col])

    for p_idx, phase_name in enumerate(phase_list):
        sig = clips[phase_name]["data"][col]
        freqs, psd = signal.welch(sig, fs=SFREQ_OUT, nperseg=512)
        mask = freqs <= 45
        ax.semilogy(freqs[mask], psd[mask],
                    color=phase_colors[p_idx],
                    linewidth=1.2,
                    label=phase_labels[p_idx] if col == 0 else "")

    # Shade frequency bands
    band_colors = ["#E8F4FD", "#FEF9E7", "#EAFAF1", "#FDEDEC", "#F5EEF8"]
    for (band, (lo, hi)), bc in zip(BANDS.items(), band_colors):
        ax.axvspan(lo, hi, alpha=0.2, color=bc)
        if col == 0:
            ax.text((lo+hi)/2, ax.get_ylim()[0] if ax.get_ylim()[0] > 0
                    else 1e-3, band, ha="center", fontsize=7, color="gray")

    ax.set_title(f"{ch_name} — Power Spectrum",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    if col == 0:
        ax.set_ylabel("Power (µV²/Hz)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 45)

# Section 3: Band Power Comparison
gs_bot = gridspec.GridSpec(
    1, n_ch,
    top=0.26, bottom=0.06,
    hspace=0.3, wspace=0.3
)

band_names = list(BANDS.keys())
x          = np.arange(len(band_names))
width      = 0.25

for col, ch_name in enumerate(ch_list):
    ax = fig.add_subplot(gs_bot[0, col])

    for p_idx, phase_name in enumerate(phase_list):
        sig = clips[phase_name]["data"][col]
        bp, _, _ = compute_band_powers(sig, SFREQ_OUT)
        values = [bp[b] for b in band_names]
        bars = ax.bar(x + p_idx * width, values, width,
                      color=phase_colors[p_idx], alpha=0.85,
                      label=phase_labels[p_idx] if col == 0 else "")

    ax.set_title(f"{ch_name} — Band Powers",
                 fontsize=9, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([b[:3] for b in band_names], fontsize=8)
    ax.set_ylabel("Log Power", fontsize=8)
    if col == 0:
        ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)

plt.savefig("seizure_closeup_PN07.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to seizure_closeup_PN07.png")
print("\nWhat to look for:")
print("  Raw EEG    — dramatic amplitude increase in ictal phase")
print("  Spectrum   — power shifts toward lower frequencies ictal")
print("  Band powers— delta/theta rise, alpha drops in preictal")