# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate virtualenv (from project root one level up)
source ../venv/Scripts/activate   # Windows bash
# or
..\venv\Scripts\activate          # Windows cmd

pip install -r requirements.txt
```

The dataset is not in the repo. Download from PhysioNet and place at:
```
data/siena-scalp-eeg-database-1.0.0/PN00/, PN01/, … PN13/
```

## Running the Pipeline

All scripts must be run from inside `preictal_classifier/` so relative paths resolve correctly.

```bash
cd preictal_classifier

# Inspect a raw EDF file
python 01_explore_dataset.py

# Full pipeline (run in order)
python scripts/03_preprocess.py          # → data/processed/PNxx.npz
python scripts/04_extract_features.py    # → data/features/features.npz
python scripts/05_train_model.py         # → models/lopo_results.json

# Optional: add coherence features before training
python scripts_v2/06_add_coherence.py    # → data/features/features_coherence.npz

# Visualize
python scripts/02_visualize_dataset.py
python scripts/06_visualize_eeg_phases.py

# Real-time hardware stream (OpenBCI Ganglion)
python connection/ganglion_stream.py --port COM3
python connection/ganglion_stream.py --native --output data.csv
```

## Architecture

### Pipeline Data Flow

```
EDF files (512 Hz, 10-20 system)
  └─ 03_preprocess.py
       ├─ channel select (8 electrodes from TARGET_CHANNELS)
       ├─ bandpass filter 0.5–40 Hz (IIR)
       ├─ resample → 250 Hz (matches Cyton hardware target)
       ├─ sliding windows: 30s window / 5s step → 83% overlap
       ├─ label: 1=preictal (0–5 min pre-onset), 0=interictal (>30 min), -1=discard
       └─ per-window z-score normalization
       → data/processed/PNxx.npz  (X: n_windows × n_ch × 7500, y: n_windows)

  └─ 04_extract_features.py / 06_add_coherence.py
       ├─ Welch PSD (nperseg=512) per window
       ├─ 64 features: 40 band powers + 16 band ratios + 8 spectral entropy
       ├─ coherence variant adds 140 features (28 channel pairs × 5 bands) → 204 total
       └─ patient-relative z-score normalization (interictal baseline only)
       → data/features/features.npz  or  features_coherence.npz

  └─ 05_train_model.py
       ├─ LOPO cross-validation (Leave-One-Patient-Out, 14 folds)
       ├─ StandardScaler fit on training fold only
       ├─ GradientBoosting (300 trees, sample_weight for imbalance)
       ├─ Dense NN: 128→64→32→1, BatchNorm+Dropout, EarlyStopping on val_auc
       └─ Decision threshold: 0.65
       → models/lopo_results.json, models/gb_PNxx.pkl, models/nn_PNxx.keras
```

### Two Montages

| Folder | Channels | Purpose |
|---|---|---|
| `scripts_v1/` | T3, T5, O1, Pz, O2, T6, T4 (7-ch) | Headband-friendly, no frontal electrodes |
| `scripts_v2/` | F7, T3, T5, C3, F8, T4, T6, C4 (8-ch) | Best LOPO generalization |
| `scripts/` | F7, T3, T5, C3, F8, T4, T6, C4 (6–8 ch) | Current active version |

### Class Imbalance (Critical)

- Raw ratio: ~169:1 (interictal:preictal)
- After windowing: ~33.7:1
- Handled via `sample_weight` (GB) and `class_weight` (NN) — never by undersampling
- **Never use accuracy as a metric.** Always report AUC-ROC, F1, Precision, Recall.
- PN01 and PN11 have zero preictal windows and are skipped during LOPO evaluation.

### Patient-Relative Normalization (Key Design Decision)

Features are z-scored against each patient's own interictal mean/std *before* pooling across patients. This removes inter-patient amplitude differences. The baseline is computed from interictal windows only so preictal deviations are preserved.

### Hardware Target

OpenBCI Cyton: 8 channels, 250 Hz, 24-bit. The 512 Hz dataset is resampled to 250 Hz to match this. `connection/ganglion_stream.py` streams from OpenBCI Ganglion (4-channel) via BrainFlow and is separate from the training pipeline.
