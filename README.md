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

## Project Structure

```
preictal_classifier/
├── scripts_v1/                  ← 7-channel headband friendly montage
│   ├── 02_visualize_dataset.py
│   ├── 03_preprocess.py
│   ├── 03_timeline_chart.py
│   ├── 04_extract_features.py
│   ├── 05_train_model.py
│   └── 06_visualize_eeg_phases.py
├── scripts_v2/                  ← 8-channel best ML performance montage
│   ├── 02_visualize_dataset.py
│   ├── 03_preprocess.py
│   ├── 03_timeline_chart.py
│   ├── 04_extract_features.py
│   ├── 05_train_model.py
│   ├── 06_add_coherence.py
│   └── 06_visualize_eeg_phases.py
├── data/                        ← gitignored
│   ├── siena-scalp-eeg-database-1.0.0/
│   ├── processed/
│   └── features/
├── models/                      ← gitignored
├── plots/                       ← gitignored
├── requirements.txt
└── README.md
```

## Channel Montages

### scripts_v1 — Headband-Friendly (7 channels)
```
T3, T5, O1, Pz, O2, T6, T4
```
Optimized for a headband that wraps around the back of the head. All electrodes are accessible without frontal placement.

### scripts_v2 — Best ML Performance (8 channels)
```
F7, T3, T5, C3, F8, T4, T6, C4
```
Bilateral temporal + frontal-temporal + central montage. Achieves best cross-patient generalization in LOPO evaluation.

## Pipeline

Each scripts folder runs in this order:

```
03_preprocess.py          ← Load EDF → filter → resample → window → label
04_extract_features.py    ← Extract 64 band power features per window
05_train_model.py         ← LOPO training: GradientBoosting + Neural Network
06_visualize_eeg_phases.py ← Visualize EEG across seizure phases
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

## Feature Extraction

64 features per window, normalized relative to each patient's own interictal baseline:

- **Band powers** (40): log power in delta, theta, alpha, beta, gamma × 8 channels
- **Band ratios** (16): theta/alpha and delta/beta × 8 channels
- **Spectral entropy** (8): 1 per channel

Patient-relative normalization is critical — features are z-scored against each patient's own interictal mean/std so the model learns deviations from *that person's* normal brain state.

## Model & Evaluation

- **Architecture**: Dense neural network (128→64→32→1) + GradientBoosting baseline
- **Validation**: Leave-One-Patient-Out (LOPO) — train on 13 patients, test on 14th
- **Class imbalance**: Handled via sample weights (33.7:1 ratio)
- **Decision threshold**: 0.65 (tuned for high precision)
- **Metrics**: AUC-ROC, F1, Precision, Recall (never accuracy)

## Results (scripts_v2, best run)

| Model | Mean AUC | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|
| Neural Network | 0.584 | 0.087 | 0.082 | 0.205 |
| GradientBoosting | 0.514 | 0.006 | 0.024 | 0.004 |

Best individual patient: PN07 — Neural Network AUC **0.884**

> Note: Results vary between runs due to neural network random initialization. AUC ~0.58 is the consistent baseline.

## Class Imbalance

```
Total recording time : 7,704 minutes (128.4 hours)
Total seizure time   :    41.7 minutes
Imbalance ratio      :   ~169:1 (raw)
After windowing      :    33.7:1
```

Never use accuracy as a metric — a model predicting "no seizure" always would achieve 97%+ accuracy.

## Known Limitations

- 14 patients is a small dataset for cross patient generalization
- PN01 and PN11 have no preictal windows (seizures occur too early in recordings)
- Some patients show AUC below 0.5 (worse than random) — their preictal signatures don't match other patients
- GradientBoosting consistently underperforms the neural network on this task

## Citation

If using the Siena dataset:
```
Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/5d4a-j060
```
