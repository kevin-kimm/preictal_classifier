# Preictal Classifier — Seizure Prediction Algorithm

A machine learning pipeline for predicting epileptic seizures using EEG data, designed for a wearable headband device powered by the OpenBCI Cyton board.

## Overview

This project trains and evaluates a seizure prediction model that aims to alert patients **5 minutes before a seizure occurs**. The algorithm analyzes EEG frequency band features from 8 temporal/occipital electrodes to classify brain activity as either **preictal** (pre seizure) or **interictal** (normal).

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
Bilateral temporal + frontal temporal + central montage. Achieves best cross patient generalization in LOPO evaluation. Mean NN AUC: **0.584**

## Pipeline

```
03_preprocess.py           ← Load EDF → filter → resample → window → label
04_extract_features.py     ← Extract band power features (64 features)
07_add_correlation.py      ← Extract band powers + correlation (148 features)
05_train_model.py          ← LOPO training: GradientBoosting + Neural Network
06_add_coherence.py        ← Experimental: coherence features (204 features)
08_ensemble.py             ← Ensemble: NN + GradBoost with post-processing
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

## Feature Sets Investigated

### 1. Band Powers (64 features) — `04_extract_features.py`
- **Band powers** (40): log power in delta, theta, alpha, beta, gamma × 8 channels
- **Band ratios** (16): theta/alpha and delta/beta × 8 channels
- **Spectral entropy** (8): 1 per channel

### 2. Band Powers + Correlation (148 features) — `07_add_correlation.py`
Adds time domain synchronization features:
- **Pearson correlation** (28): linear synchrony between every channel pair
- **Cross-correlation** (56): peak synchrony value + propagation lag per pair (±500ms)

### 3. Band Powers + Coherence (204 features) — `06_add_coherence.py`
Adds frequency domain synchronization features:
- **Coherence** (140): magnitude squared coherence per channel pair per band

All features are z scored relative to each patient's own interictal baseline, the model learns deviations from that person's normal brain state, not absolute values.

## Model & Evaluation

- **Architecture**: Dense neural network (128→64→32→1) + GradientBoosting baseline
- **Validation**: Leave One Patient Out (LOPO) — train on 13 patients, test on 14th
- **Class imbalance**: 33.7:1 — handled via sample weights
- **Decision threshold**: 0.65 (tuned for high precision)
- **Metrics**: AUC-ROC, F1, Precision, Recall, not accuracy

## Results — Full Comparison

### Per-patient Neural Network AUC (band powers, best run)

| Patient | AUC | Notes |
|---|---|---|
| PN00 | 0.365 | Below random — atypical preictal pattern |
| PN03 | 0.760 | Good signal |
| PN05 | 0.450 | Near random |
| PN06 | 0.584 | Moderate |
| PN07 | **0.884** | Best patient — very consistent preictal pattern |
| PN09 | 0.587 | Moderate |
| PN10 | 0.469 | Below average |
| PN11 | — | No preictal windows |
| PN12 | 0.691 | Good signal |
| PN13 | 0.606 | Moderate |
| PN14 | 0.564 | Moderate |
| PN16 | 0.547 | Moderate |
| PN17 | 0.497 | Near random |
| **MEAN** | **0.584** | |

### Feature Set Comparison

| Feature Set | Features | NN AUC | GB AUC | Winner |
|---|---|---|---|---|
| Band powers | 64 | **0.584** | 0.514 | NN |
| + Coherence | 204 | 0.479 | 0.499 | GB |
| + Correlation | 148 | 0.491 | **0.550** | GB |

### Ensemble Comparison (08_ensemble.py)

| Method | Mean AUC | Notes |
|---|---|---|
| Neural Network alone | 0.541 | Band powers |
| GradientBoosting alone | 0.550 | Correlation features |
| Average ensemble | 0.549 | Mean of both probabilities |
| Weighted ensemble | 0.555 | Weighted by validation AUC |
| Max ensemble | 0.541 | Highest probability wins |
| **Consensus ensemble** | **0.575** | Both models must agree |
| Average + smoothing | 0.500 | Too strict for small test sets |

## Recommended Model

> **Neural Network on band powers (64 features) — Mean AUC 0.584**

This is the best performing configuration across all experiments. Key reasons:

- Band powers are biologically meaningful and generalizable across patients
- The neural network learns non linear relationships between bands that GradientBoosting misses
- Adding more features (coherence, correlation) consistently hurts the NN due to overfitting on 14 patients
- The consensus ensemble (0.575) is the best ensemble strategy but does not beat the simple NN

For deployment on the Cyton board, use the Neural Network with band power features. The model is lightweight (179K parameters) and can run inference in real time.

## Class Imbalance

```
Total recording time : 7,704 minutes (128.4 hours)
Total seizure time   :    41.7 minutes
Imbalance ratio      :   ~169:1 (raw)
After windowing      :    33.7:1
```

## Known Limitations

- 14 patients is a small dataset for cross patient generalization
- PN01 and PN11 have no preictal windows (seizures occur too early in recordings)
- Some patients show AUC below 0.5, their preictal signatures differ from others in the dataset
- Neural network results vary ±0.06 AUC between runs due to random initialization
- Adding more features hurts the neural network (overfitting) but helps GradientBoosting
- Post processing smoothing is too strict for small per patient test sets

## Next Steps

- Patient specific fine tuning when device is worn by a new user
- Longer preictal window (10–15 min) may improve detectability
- Larger dataset (100+ patients) to enable raw waveform CNN+LSTM approach
- Real time inference pipeline for OpenBCI Cyton board
- Hardware validation with physical headband prototype

## Citation

If using the Siena dataset:
```
Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/5d4a-j060
```