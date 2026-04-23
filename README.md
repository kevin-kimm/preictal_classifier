# Preictal Classifier — Seizure Prediction Algorithm

A machine learning pipeline for predicting epileptic seizures using EEG data, designed for a wearable headband device powered by the OpenBCI Cyton board.

## Overview

This project trains and evaluates a seizure prediction model that aims to alert patients **5 minutes before a seizure occurs**. The algorithm analyzes EEG frequency band features from 8 temporal/occipital electrodes to classify brain activity as either **preictal** (pre seizure) or **interictal** (normal).

## Hardware Target

- **Board**: [OpenBCI Cyton](https://shop.openbci.com/products/cyton-biosensing-board-8-channel) — 8-channel, 250 Hz, 24-bit resolution
- **Form factor**: Wearable headband
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

### scripts_v3 — Headband + Rich Features (7 channels)
```
T3, T5, O1, Pz, O2, T6, T4
```
Same headband montage as v1 but with significantly richer features (287 vs 64). Adds artifact rejection, common average reference, notch filter, and postictal discard window. Mean NN AUC: **0.541**, Mean GB AUC: **0.554**

## Pipeline

```
03_preprocess.py           ← Load EDF → filter → resample → window → label
04_extract_features.py     ← Extract band power features (64 features)
07_add_correlation.py      ← Extract band powers + correlation (148 features)
05_train_model.py          ← LOPO training: GradientBoosting + Neural Network
06_add_coherence.py        ← Experimental: coherence features (204 features)
08_ensemble.py             ← Ensemble: NN + GradBoost with post-processing
06_visualize_eeg_phases.py ← Visualize EEG across seizure phases
09b_seizure_closeup_PN07.py← Best patient seizure close-up (AUC 0.914)
09b_seizure_closeup_PN00.py← Worst patient seizure close-up (AUC 0.294)
10_seizure_transition.py   ← 30s before → 30s during seizure transition
11_best_patients_transition← Transition plot for top 3 patients
12_patient_review.py       ← Full review across all 14 patients and versions
```

## Preprocessing Details

| Parameter | Value |
|---|---|
| Sample rate | 250 Hz (resampled from 512 Hz to match Cyton) |
| Window size | 30 seconds (7,500 samples) |
| Step size | 5 seconds (83% overlap) |
| Bandpass filter | 0.5–40 Hz |
| Notch filter | 50 Hz (v3 only) |
| Artifact rejection | Peak-to-peak > 500µV discarded (v3 only) |
| Reference | Common average (v3 only) |
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

### 4. Full Feature Set (287 features) — `scripts_v3/04_extract_features.py`
- Band powers + ratios + spectral entropy (56)
- **Hjorth parameters** (14): mobility and complexity per channel
- **Coherence** (105): 21 pairs × 5 bands
- **Phase-Locking Value / PLV** (105): phase synchrony independent of amplitude
- **Permutation entropy** (7): signal complexity per channel

All features are z scored relative to each patient's own interictal baseline — the model learns deviations from that person's normal brain state, not absolute values.

## Model & Evaluation

- **Architecture**: Dense neural network (128→64→32→1) + GradientBoosting baseline
- **v3 improvement**: Larger NN (256→128→64→32→1) with focal loss
- **Validation**: Leave One Patient Out (LOPO) — train on 13 patients, test on 14th
- **Class imbalance**: 33.7:1 — handled via sample weights
- **Decision threshold**: 0.65 (tuned for high precision)
- **Metrics**: AUC-ROC, F1, Precision, Recall — never accuracy

## Results — Full Comparison

### Per-Patient Best AUC (across all versions)

| Patient | Age | Type | Best AUC | Grade | Best Model |
|---|---|---|---|---|---|
| PN07 | 20 | IAS | **0.914** | ✅ Predictable | v2 NN |
| PN03 | 54 | IAS | **0.849** | ✅ Predictable | v3 NN |
| PN16 | 41 | IAS | **0.823** | ✅ Predictable | v3 GB |
| PN13 | 34 | IAS | **0.754** | ✅ Predictable | v3 NN |
| PN12 | 71 | IAS | **0.717** | ✅ Predictable | v3 NN |
| PN06 | 36 | IAS | **0.707** | ✅ Predictable | v3 NN |
| PN10 | 25 | FBTC | 0.646 | ⚠️ Modest | v3 GB |
| PN17 | 42 | IAS | 0.619 | ⚠️ Modest | v2 NN |
| PN09 | 27 | IAS | 0.610 | ⚠️ Modest | v2 NN |
| PN14 | 49 | WIAS | 0.577 | ⚠️ Modest | v2 NN |
| PN00 | 55 | IAS | 0.532 | ❌ Poor | v3 GB |
| PN05 | 51 | IAS | 0.463 | ❌ Poor | v2 GB |
| PN01 | 46 | IAS | — | No preictal data | — |
| PN11 | 58 | IAS | — | No preictal data | — |

**6 predictable, 4 modest, 2 poor, 2 no preictal data**

### Pipeline Version Comparison

| Version | Channels | Features | NN AUC | GB AUC |
|---|---|---|---|---|
| scripts_v1 | 7ch headband | 64 | 0.563 | 0.514 |
| **scripts_v2** | **8ch best ML** | **64** | **0.584** | 0.514 |
| scripts_v2 | 8ch best ML | 148 (+corr) | 0.491 | 0.550 |
| scripts_v2 | 8ch best ML | 204 (+coh) | 0.479 | 0.499 |
| scripts_v3 | 7ch headband | 287 (full) | 0.541 | 0.554 |

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

> **Neural Network on band powers (64 features), 8-channel montage — Mean AUC 0.584**

Key reasons:
- Band powers are biologically meaningful and generalizable across patients
- The neural network learns non linear relationships between bands that GradientBoosting misses
- Adding more features consistently hurts the NN due to overfitting on 14 patients
- The consensus ensemble (0.575) is the best ensemble strategy but does not beat the simple NN
- Lightweight — 179K parameters, runs inference in real time

For the physical headband, use **scripts_v3** with the 7-channel posterior montage — comparable results (0.541 NN, 0.554 GB) while matching electrode placement constraints.

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
- Some patients show AUC below 0.5 — their preictal signatures differ from others
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
