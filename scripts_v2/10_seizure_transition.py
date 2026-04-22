"""
=============================================================
  Siena Scalp EEG — Seizure Transition Visualization
  10_seizure_transition.py

  Shows a continuous 60-second window spanning the seizure
  onset for both PN07 (predictable) and PN00 (unpredictable):
    - First 30 seconds: before seizure (preictal)
    - Last 30 seconds:  during seizure (ictal)

  A vertical red line marks the exact seizure onset.

  This shows the REAL-TIME transition — exactly what the
  device would see as a seizure begins.
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

SFREQ_OUT     = 250
CHANNELS      = ["T3", "T4", "F7", "F8"]
BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 40.0

# Patients to visualize
PATIENTS = {
    "PN07": {
        "file":   "PN07-1.edf",
        "onset":  367.65 * 60,   # seconds into file
        "auc":    0.884,
        "label":  "Predictable patient",
    },
    "PN00": {
        "file":   "PN00-1.edf",
        "onset":  1143.0,
        "auc":    0.294,
        "label":  "Unpredictable patient",
    },
}

WINDOW_BEFORE = 30   # seconds before onset to show
WINDOW_AFTER  = 30   # seconds after onset to show


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def find_channel(ch_names, target):
    for ch in ch_names:
        clean = re.sub(r"^EEG\s*", "", ch).strip()
        clean = re.split(r"[-–]", clean)[0].strip()
        if clean.upper() == target.upper():
            return ch
    return None


# ─────────────────────────────────────────────────────────────
# FIGURE — 2 patients side by side, 4 channels each
# ─────────────────────────────────────────────────────────────
n_channels = len(CHANNELS)
n_patients = len(PATIENTS)

fig, axes = plt.subplots(
    n_channels, n_patients,
    figsize=(18, 10),
    sharey=False,
    squeeze=False
)

fig.suptitle(
    "EEG Seizure Transition — 30 Seconds Before → 30 Seconds During\n"
    "Red line = seizure onset",
    fontsize=14, fontweight="bold"
)

for col, (patient_id, info) in enumerate(PATIENTS.items()):
    edf_path = DATA_ROOT / patient_id / info["file"]
    print(f"\nLoading {patient_id} — {info['file']}...")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.resample(SFREQ_OUT, verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="iir", verbose=False)

    # Find channels
    ch_map = {}
    for target in CHANNELS:
        found = find_channel(raw.ch_names, target)
        if found:
            ch_map[target] = found

    raw.pick(list(ch_map.values()))
    data, _ = raw[:]
    data_uv  = data * 1e6
    total_dur = data.shape[1] / SFREQ_OUT

    # Extract 60-second window centered on onset
    onset    = info["onset"]
    clip_start = max(0, onset - WINDOW_BEFORE)
    clip_end   = min(total_dur - 1, onset + WINDOW_AFTER)

    s_idx = int(clip_start * SFREQ_OUT)
    e_idx = int(clip_end   * SFREQ_OUT)

    clip_data = data_uv[:, s_idx:e_idx]
    clip_time = np.linspace(
        -WINDOW_BEFORE,
        WINDOW_AFTER,
        e_idx - s_idx
    )

    # Onset position in clip time = 0
    onset_in_clip = 0.0

    print(f"  Clip: {clip_start:.0f}s – {clip_end:.0f}s "
          f"({clip_end-clip_start:.0f}s total)")

    for row, ch_name in enumerate(CHANNELS):
        ax = axes[row, col]

        if ch_name not in ch_map:
            ax.text(0.5, 0.5, f"{ch_name} not found",
                    transform=ax.transAxes, ha="center")
            continue

        ch_idx = list(ch_map.keys()).index(ch_name)
        sig    = clip_data[ch_idx]

        # Color the background — blue before, red after onset
        ax.axvspan(-WINDOW_BEFORE, 0,
                   color="#4A90D9", alpha=0.08, zorder=0)
        ax.axvspan(0, WINDOW_AFTER,
                   color="#D0021B", alpha=0.08, zorder=0)

        # Plot signal — color changes at onset
        before_mask = clip_time <= 0
        after_mask  = clip_time >= 0

        ax.plot(clip_time[before_mask], sig[before_mask],
                color="#2171B5", linewidth=0.7, alpha=0.9,
                label="Pre-seizure" if row == 0 and col == 0 else "")
        ax.plot(clip_time[after_mask], sig[after_mask],
                color="#CB181D", linewidth=0.7, alpha=0.9,
                label="Seizure" if row == 0 and col == 0 else "")

        # Seizure onset line
        ax.axvline(0, color="red", linewidth=2,
                   linestyle="--", zorder=5,
                   label="Seizure onset" if row == 0 and col == 0 else "")

        # Labels
        if row == 0:
            ax.set_title(
                f"{patient_id} — {info['label']}\n"
                f"LOPO AUC: {info['auc']}",
                fontsize=11, fontweight="bold",
                color="#2171B5" if info["auc"] > 0.5 else "#CB181D"
            )
        if col == 0:
            ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=9,
                          rotation=0, labelpad=45, va="center")
        if row == n_channels - 1:
            ax.set_xlabel("Time relative to seizure onset (seconds)",
                          fontsize=9)

        # Y limits
        amp  = np.percentile(np.abs(sig), 98)
        ylim = max(amp * 1.4, 30)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(-WINDOW_BEFORE, WINDOW_AFTER)

        # Zero line
        ax.axhline(0, color="gray", linewidth=0.3, linestyle=":")

        # Amplitude annotation
        ax.text(0.99, 0.95, f"±{amp:.0f}µV",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray")

        # X axis labels
        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
        ax.set_xticklabels(["-30s", "-20s", "-10s", "ONSET",
                            "+10s", "+20s", "+30s"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Shade label
        if row == 0:
            ax.text(-15, ylim * 0.85, "Pre-seizure",
                    ha="center", fontsize=9,
                    color="#2171B5", fontweight="bold")
            ax.text(15, ylim * 0.85, "Seizure",
                    ha="center", fontsize=9,
                    color="#CB181D", fontweight="bold")

# Legend
pre_patch    = mpatches.Patch(color="#2171B5", alpha=0.7,
                               label="Pre-seizure (30s before)")
ictal_patch  = mpatches.Patch(color="#CB181D", alpha=0.7,
                               label="Ictal (30s after onset)")
onset_line   = plt.Line2D([0], [0], color="red",
                           linewidth=2, linestyle="--",
                           label="Seizure onset")
fig.legend(
    handles=[pre_patch, ictal_patch, onset_line],
    loc="lower center", ncol=3, fontsize=10,
    bbox_to_anchor=(0.5, -0.02), frameon=True
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("seizure_transition.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Saved to seizure_transition.png")
print("\nWhat to look for:")
print("  PN07 — dramatic change AT the onset line")
print("  PN00 — subtle or no change at the onset line")
print("  This is the moment the device needs to predict 5 min early")