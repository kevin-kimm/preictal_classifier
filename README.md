# Preictal Classifier — Seizure Prediction Algorithm

A machine learning pipeline for predicting epileptic seizures using EEG data, designed for deployment on a wearable headband device powered by the OpenBCI Cyton board.

## Overview

This project trains and evaluates a seizure prediction model that aims to alert patients **5 minutes before a seizure occurs**. The algorithm analyzes EEG frequency band features from 8 temporal/occipital electrodes to classify brain activity as either **preictal** (pre-seizure) or **interictal** (normal).

## Hardware Target

- **Board**: [OpenBCI Cyton](https://shop.openbci.com/products/cyton-biosensing-board-8-channel) — 8-channel, 250 Hz, 24-bit resolution
- **Electrode positions**: T3, T5, O1, O2, T6, T4, F7, F8

## Dataset

[Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/) — Detti, P. (2020)

- 14 patients with epilepsy
- 128 hours of EEG recording
- 47 seizures total
- 512 Hz sample rate, 10-20 electrode system
- Seizure types: IAS (86%), FBTC (7%), WIAS (7%)

> The dataset is not included in this repository. Download it from PhysioNet and place it in `data/siena-scalp-eeg-database-1.0.0/`.


## Channel Montages

### scripts_v1 — Headband-Friendly (7 channels)
```
T3, T5, O1, Pz, O2, T6, T4
```
Optimized for a headband that wraps around the back of the head. All electrodes are accessible without frontal placement. Mean NN AUC: **0.563**

### scripts_v2 — Best ML Performance (8 channels)
```
F7, T3, T5, C3, F8, T4, T6, C4
```
Bilateral temporal + frontal-temporal + central montage. Achieves best cross-patient generalization in LOPO evaluation. Mean NN AUC: **0.584**

## Pipeline

Each scripts folder runs in this order:

```
03_preprocess.py           ← Load EDF → filter → resample → window → label
04_extract_features.py     ← Extract band power features (64)
07_add_correlation.py      ← Extract band powers + correlation (148) [v2 only]
05_train_model.py          ← LOPO training: GradientBoosting + Neural Network
06_visualize_eeg_phases.py ← Visualize EEG across seizure phases
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/kevin-kimm/preictal_classifier.git
cd preictal_classifier

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the Siena dataset from PhysioNet
# Place in: data/siena-scalp-eeg-database-1.0.0/

# 5. Run the pipeline (example using scripts_v2)
python3 scripts_v2/03_preprocess.py
python3 scripts_v2/04_extract_features.py
python3 scripts_v2/05_train_model.py
```

## Preprocessing Details

| Parameter | Value |
|---|---|
| Sample rate | 250 Hz (resampled from 512 Hz to match Cyton) |
| Window size | 30 seconds (7,500 samples) |
| Step size | 5 seconds (83% overlap) |
| Bandpass filter | 0.5–40 Hz |
| Preictal label | 0–5 min before seizure onset |
| Buffer zone | 5–30 min before seizure (discarded) |
| Interictal label | >30 min from any seizure |

## Feature Sets

### Band Powers (64 features) — `04_extract_features.py`
Normalized relative to each patient's own interictal baseline:
- **Band powers** (40): log power in delta, theta, alpha, beta, gamma × 8 channels
- **Band ratios** (16): theta/alpha and delta/beta × 8 channels
- **Spectral entropy** (8): 1 per channel

### Band Powers + Correlation (148 features) — `07_add_correlation.py`
Adds time-domain synchronization features on top of band powers:
- **Pearson correlation** (28): linear synchrony between every channel pair
- **Cross-correlation** (56): peak synchrony value + propagation lag per pair (±500ms)

Patient-relative normalization is applied to all feature sets — features are
z-scored against each patient's own interictal mean/std so the model learns
deviations from that person's normal brain state, not absolute values.

## Model & Evaluation

- **Architecture**: Dense neural network (128→64→32→1) + GradientBoosting baseline
- **Validation**: Leave One Patient Out (LOPO) — train on 13 patients, test on 14th
- **Class imbalance**: Handled via sample weights (33.7:1 ratio)
- **Decision threshold**: 0.65 (tuned for high precision)
- **Metrics**: AUC-ROC, F1, Precision, Recall (never accuracy)

## Results Summary

### Band Powers only (64 features)
| Model | Mean AUC | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|
| Neural Network | **0.584** | 0.087 | 0.082 | 0.205 |
| GradientBoosting | 0.514 | 0.006 | 0.024 | 0.004 |

### Band Powers + Correlation (148 features)
| Model | Mean AUC | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|
| Neural Network | 0.491 | 0.060 | 0.061 | 0.089 |
| GradientBoosting | **0.550** | 0.004 | 0.052 | 0.002 |

**Key finding**: Each model has a different optimal feature set.
- Neural Network performs best with band powers only (64 features)
- GradientBoosting performs best with correlation features added (148 features)
- Best individual result: PN07 Neural Network AUC **0.884** (band powers)

## Class Imbalance

```
Total recording time : 7,704 minutes (128.4 hours)
Total seizure time   :    41.7 minutes
Imbalance ratio      :   ~169:1 (raw)
After windowing      :    33.7:1
```

Never use accuracy as a metric — a model predicting "no seizure" always
would achieve 97%+ accuracy. Always evaluate with AUC, F1, Precision, Recall.

## Known Limitations

- 14 patients is a small dataset for cross-patient generalization
- PN01 and PN11 have no preictal windows (seizures occur too early in recordings)
- Some patients show AUC below 0.5 — their preictal signatures differ from others
- Neural network results vary between runs due to random initialization (~±0.06 AUC)
- Adding more features (coherence, correlation) helps GradientBoosting but hurts the neural network due to overfitting on limited data

## Next Steps

- Ensemble model combining GradientBoosting (148 features) + Neural Network (64 features)
- Post-processing smoothing — require 3 consecutive positive windows before alerting
- Per-patient threshold tuning instead of fixed 0.65
- Real-time inference pipeline for OpenBCI Cyton board

## Citation

If using the Siena dataset:
```
Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/5d4a-j060
```
