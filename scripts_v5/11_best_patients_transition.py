"""
=============================================================
  Siena Scalp EEG — Seizure Transition for Best Patients
  11_best_patients_transition.py

  Shows the 30s before → 30s during transition for the
  3 best performing patients:
    PN07  AUC 0.884
    PN03  AUC 0.760
    PN12  AUC 0.691

  One row per patient, 4 channels per row.
  Red dashed line = seizure onset.
=============================================================
"""

import re
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT     = Path("data/siena-scalp-eeg-database-1.0.0")
SFREQ_OUT     = 250
CHANNELS      = ["T3", "T4", "F7", "F8"]
BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 40.0
WINDOW_BEFORE = 30
WINDOW_AFTER  = 30

# Best 3 patients with seizure info 
# Onset times in seconds from start of the EDF file used
PATIENTS = {
    "PN07": {
        "file":  "PN07-1.edf",
        "onset": 367.65 * 60,
        "auc":   0.884,
    },
    "PN03": {
        "file":  "PN03-1.edf",
        "onset": 644.55 * 60,
        "auc":   0.760,
    },
    "PN12": {
        "file":  "PN12-4.edf",
        "onset": 163.53 * 60,
        "auc":   0.691,
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


# Figure — 3 patients × 4 channels
n_patients = len(PATIENTS)
n_channels = len(CHANNELS)

fig, axes = plt.subplots(
    n_patients, n_channels,
    figsize=(20, 12),
    sharey=False,
    squeeze=False
)

fig.suptitle(
    "EEG Seizure Transition — Best 3 Patients\n"
    "30 Seconds Before → 30 Seconds During Seizure  |  "
    "Red dashed line = seizure onset",
    fontsize=14, fontweight="bold"
)

for row, (patient_id, info) in enumerate(PATIENTS.items()):
    edf_path = DATA_ROOT / patient_id / info["file"]
    print(f"Loading {patient_id} — {info['file']}...")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True,
                               verbose=False)
    raw.resample(SFREQ_OUT, verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH,
               method="iir", verbose=False)

    # Find channels
    ch_map = {}
    for target in CHANNELS:
        found = find_channel(raw.ch_names, target)
        if found:
            ch_map[target] = found

    raw.pick(list(ch_map.values()))
    data, _   = raw[:]
    data_uv   = data * 1e6
    total_dur = data.shape[1] / SFREQ_OUT
    onset     = info["onset"]

    # Extract 60s clip
    clip_start = max(0, onset - WINDOW_BEFORE)
    clip_end   = min(total_dur - 1, onset + WINDOW_AFTER)
    s_idx      = int(clip_start * SFREQ_OUT)
    e_idx      = int(clip_end   * SFREQ_OUT)

    clip_data  = data_uv[:, s_idx:e_idx]
    clip_time  = np.linspace(-WINDOW_BEFORE, WINDOW_AFTER,
                              e_idx - s_idx)

    print(f"  Onset: {onset/60:.1f} min | "
          f"Clip: {clip_start:.0f}s–{clip_end:.0f}s")

    for col, ch_name in enumerate(CHANNELS):
        ax = axes[row, col]

        if ch_name not in ch_map:
            ax.text(0.5, 0.5, f"{ch_name}\nnot found",
                    transform=ax.transAxes, ha="center",
                    fontsize=9, color="gray")
            ax.set_visible(True)
            continue

        ch_idx = list(ch_map.keys()).index(ch_name)
        sig    = clip_data[ch_idx]

        # Background shading
        ax.axvspan(-WINDOW_BEFORE, 0,
                   color="#4A90D9", alpha=0.08, zorder=0)
        ax.axvspan(0, WINDOW_AFTER,
                   color="#D0021B", alpha=0.08, zorder=0)

        # Signal
        before_mask = clip_time <= 0
        after_mask  = clip_time >= 0
        ax.plot(clip_time[before_mask], sig[before_mask],
                color="#2171B5", linewidth=0.7, alpha=0.95)
        ax.plot(clip_time[after_mask], sig[after_mask],
                color="#CB181D", linewidth=0.7, alpha=0.95)

        # Onset line
        ax.axvline(0, color="red", linewidth=2,
                   linestyle="--", zorder=5)

        # Labels
        if row == 0:
            ax.set_title(f"{ch_name}", fontsize=11,
                         fontweight="bold")
        if col == 0:
            ax.set_ylabel(
                f"{patient_id}\nAUC {info['auc']}\n\n"
                f"{ch_name} (µV)",
                fontsize=9, rotation=0,
                labelpad=60, va="center"
            )
        if row == n_patients - 1:
            ax.set_xlabel(
                "Time relative to seizure onset (s)",
                fontsize=8)

        # Y limits
        amp  = np.percentile(np.abs(sig), 98)
        ylim = max(amp * 1.4, 30)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(-WINDOW_BEFORE, WINDOW_AFTER)
        ax.axhline(0, color="gray", linewidth=0.3, linestyle=":")

        # Amplitude label
        ax.text(0.99, 0.95, f"±{amp:.0f}µV",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray")

        # Phase labels on first row only
        if row == 0:
            ax.text(-15, ylim * 0.82, "Pre-seizure",
                    ha="center", fontsize=8,
                    color="#2171B5", fontweight="bold")
            ax.text(15, ylim * 0.82, "Seizure",
                    ha="center", fontsize=8,
                    color="#CB181D", fontweight="bold")

        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
        ax.set_xticklabels(
            ["-30s", "-20s", "-10s", "ONSET",
             "+10s", "+20s", "+30s"], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

# Legend
pre_patch  = mpatches.Patch(color="#2171B5", alpha=0.7,
                             label="Pre-seizure (30s before)")
ict_patch  = mpatches.Patch(color="#CB181D", alpha=0.7,
                             label="Ictal (30s after onset)")
onset_line = plt.Line2D([0], [0], color="red",
                         linewidth=2, linestyle="--",
                         label="Seizure onset")
fig.legend(
    handles=[pre_patch, ict_patch, onset_line],
    loc="lower center", ncol=3, fontsize=10,
    bbox_to_anchor=(0.5, -0.01), frameon=True
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("best_patients_transition.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("\nSaved to best_patients_transition.png")