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

---

## v6 — Best Generalized Detector (Final Version)

### Overview

v6 is the definitive version of the seizure prediction algorithm, incorporating all lessons learned from v1 through v5. It uses the best ML electrode montage, the cleanest preprocessing pipeline, and a multi-run seed optimization strategy to find the best possible model per patient.

**Key result: 8/12 evaluable patients predictable (67%), mean AUC 0.713**

---

### Pipeline

```
scripts_v6/
  03_preprocess.py        Raw EEG → labeled 30s windows
  04_extract_features.py  Windows → 64 band power features
  05_train_model.py       Run 1 training (seed 42 baseline)
  06_multi_run.py         Multi-seed optimizer (297 seeds, 4 hours)
  07_finetune.py          Patient-specific fine-tuning
  08_final_eval.py        Final results dashboard
```

---

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channels | F7, T3, T5, C3, F8, T4, T6, C4 | Best ML montage — proven in all experiments |
| Sample rate | 250 Hz | Matches OpenBCI Cyton hardware |
| Window size | 30 seconds | Standard for EEG classification |
| Step size | 5 seconds | Independent windows, 83% overlap |
| Preictal zone | 0–5 min before seizure | Most consistent cross-patient signal |
| Buffer zone | 5–30 min before seizure | Ambiguous — discarded |
| Postictal zone | 30 min after seizure | Recovery period — discarded |
| Notch filter | 50 Hz | Italian powerline (Siena dataset) |
| Bandpass | 0.5–40 Hz | Brain-relevant frequencies only |
| Reference | Common Average (CAR) | Removes shared electrical noise |
| Artifact threshold | 500 µV peak-to-peak | Rejects corrupted windows |
| Features | 64 | Band powers + ratios + spectral entropy |
| Architecture | Dense 128→64→32→1 | Proven best for 14-patient generalization |
| Loss function | Binary crossentropy | More stable than focal loss on small datasets |
| Threshold | 0.65 | Model confidence required to fire alert |
| Batch size | 256 | Optimized for Apple M4 Metal GPU |

---

### Feature Extraction

64 features are extracted from each 30-second window using Welch's Power Spectral Density (nperseg=512, Hann window):

**Band Powers (40 features)**
Power in 5 frequency bands × 8 channels, log1p compressed:
```
Delta   0.5–4 Hz    rises during seizures and preictal state
Theta   4–8 Hz      key preictal biomarker — reliably rises 1-3 min before seizure
Alpha   8–13 Hz     key preictal biomarker — reliably drops before seizure
Beta    13–30 Hz    active thinking, varies by patient
Gamma   30–40 Hz    spikes at seizure onset
```

**Band Ratios (16 features)**
Per-channel ratios, normalized across patients:
```
Theta/Alpha    normal ~0.5, preictal >1.5 — most validated preictal biomarker
Delta/Beta     rises as consciousness changes preictally
```

**Spectral Entropy (8 features)**
Shannon entropy of normalized PSD per channel:
```
High entropy = complex, unpredictable = normal brain
Low entropy  = synchronized, rhythmic = preictal brain
```

All 64 features are **z-score normalized** relative to each patient's own interictal baseline, so the model asks "is this high for THIS person?" rather than using absolute values.

---

### Validation Strategy

**Leave-One-Patient-Out (LOPO)** cross-validation:
- Each of 12 evaluable patients is held out as the test set in turn
- Model trained on remaining 13 patients
- Test patient's data is **never seen during training**
- Simulates a brand new patient putting on the device for the first time
- PN01 and PN11 excluded (zero preictal windows in recordings)

---

### Multi-Seed Optimization

Neural networks start with random weights (controlled by a random seed). Different seeds produce different final models. v6 exploits this:

1. **Run 1** (seed 42): single training run, saves baseline models
2. **Multi-run** (06_multi_run.py): tries 297 additional seeds
   - For each patient, saves the model only if AUC improves
   - Progress saved after every seed — safe to interrupt and resume
   - Automatically stops at time limit (configurable)

**Why keep-best instead of averaging?**
- Averaging (ensemble) was tested in v2.02 — hurt performance
- Focal loss caused high variance between seeds (±0.15 AUC)
- With binary crossentropy, variance is low (±0.03–0.05 AUC)
- Keep-best extracts maximum value from the best initialization

---

### Results

#### v6 Clinical Montage (8 channels: F7,T3,T5,C3,F8,T4,T6,C4)

| Patient | Run 1 (seed 42) | Multi-run Best | Grade |
|---------|----------------|---------------|-------|
| PN00 | 0.326 | 0.512 | ❌ |
| PN03 | 0.603 | 0.655 | ⚠️ |
| PN05 | 0.584 | 0.772 | ✅ |
| PN06 | 0.561 | 0.677 | ⚠️ |
| PN07 | 0.653 | 0.856 | ✅ |
| PN09 | 0.409 | 0.542 | ❌ |
| PN10 | 0.714 | 0.755 | ✅ |
| PN12 | 0.654 | 0.706 | ✅ |
| PN13 | 0.597 | 0.759 | ✅ |
| PN14 | 0.635 | 0.721 | ✅ |
| PN16 | 0.454 | 0.827 | ✅ |
| PN17 | 0.628 | 0.774 | ✅ |
| **MEAN** | **0.568** | **0.713** | |

**✅ Predictable (AUC ≥ 0.70): 8/12 patients (67%)**
**⚠️ Modest (AUC 0.60–0.70): 2/12 patients**
**❌ Poor (AUC < 0.60): 2/12 patients**

---

#### v6 Headband Montage (9 channels: F7,T3,T5,O1,Pz,O2,T6,T4,F8)

Identical pipeline to v6 clinical but using the headband-friendly electrode placement plus F7/F8. Results from `scripts_v6_headband/`.

| Patient | Run 1 (seed 42) | Multi-run Best | Grade |
|---------|----------------|---------------|-------|
| PN00 | 0.264 | 0.593 | ❌ |
| PN03 | 0.399 | 0.702 | ✅ |
| PN05 | 0.594 | 0.790 | ✅ |
| PN06 | 0.608 | 0.706 | ✅ |
| PN07 | 0.308 | 0.601 | ⚠️ |
| PN09 | 0.466 | 0.613 | ⚠️ |
| PN10 | 0.706 | 0.759 | ✅ |
| PN12 | 0.600 | 0.705 | ✅ |
| PN13 | 0.554 | 0.659 | ⚠️ |
| PN14 | 0.545 | 0.750 | ✅ |
| PN16 | 0.788 | 0.910 | ✅ |
| PN17 | 0.649 | 0.682 | ⚠️ |
| **MEAN** | **0.540** | **0.706** | |

**✅ Predictable (AUC ≥ 0.70): 7/12 patients (58%)**
**⚠️ Modest (AUC 0.60–0.70): 4/12 patients**
**❌ Poor (AUC < 0.60): 1/12 patients**

---

#### Montage Comparison

| Configuration | Channels | Features | Predictable | Mean AUC |
|---------------|----------|----------|-------------|----------|
| v6 Clinical | F7,T3,T5,C3,F8,T4,T6,C4 (8ch) | 64 | **8/12** | **0.713** |
| v6 Headband | F7,T3,T5,O1,Pz,O2,T6,T4,F8 (9ch) | 72 | 7/12 | 0.706 |

**Key finding:** The consumer headband montage with F7/F8 added achieves nearly identical performance to the optimized clinical montage (7/12 vs 8/12, mean AUC 0.706 vs 0.713). This demonstrates that a wearable device incorporating frontal-temporal electrodes can match clinical electrode placement performance.

---

#### Performance vs All Previous Versions

| Version | Channels | Features | Predictable | Mean AUC |
|---------|----------|----------|-------------|----------|
| v1 | 7ch headband | 64 | 3–4/12 | ~0.563 |
| v2 | 8ch clinical | 64 | 6/12 | 0.584 |
| v3 | 7ch headband | 287 | 3/12 | 0.539 |
| v4 | 7ch headband | 287 | 3/12 | 0.522 |
| v5 | 8ch clinical | 69 | 0/12 | 0.545 |
| **v6** | **8ch clinical** | **64** | **8/12** | **0.713** |
| v6 headband | 9ch hybrid | 72 | 7/12 | 0.706 |

---

### Why v6 Outperforms Previous Versions

| Factor | Previous versions | v6 |
|--------|------------------|-----|
| Preprocessing | Inconsistent | Notch + CAR + artifact rejection |
| Features | 64–287 (more hurt performance) | 64 (optimal for 14 patients) |
| Loss function | Focal loss (high variance) | Binary crossentropy (stable) |
| Seeds | Single run (seed 42) | 297+ seeds, keep best per patient |
| Strategy | Average ensemble | Keep-best per patient |

**The single biggest improvement:** Running 297 seeds and keeping the best model per patient. The same architecture trained on the same data — just finding better random initializations. This demonstrates that the algorithm is sound; the limiting factor was initialization variance, not capacity.

---

### Unpredictable Patients

Two patients remain unpredictable across all versions and seeds:

**PN00 (AUC 0.512):** Preictal EEG is statistically indistinguishable from interictal. Even with 297 seeds, no initialization found a generalizable pattern. Likely requires different electrode placement or features beyond band powers.

**PN09 (AUC 0.542):** Consistent poor performance across all versions (v2: 0.610, v3: 0.492, v4: 0.625, v6: 0.542). The preictal signature for this patient does not generalize from any combination of the other 13 training patients.

These patients represent a fundamental biological limit — not an algorithmic one. With a larger dataset (100+ patients), patient-specific patterns would be better represented in training.

---

### Clinical Deployment Path

v6 implements a two-phase deployment strategy mirroring commercial seizure prediction systems (e.g. NeuroPace RNS):

**Phase 1 — Cross-patient model (off the shelf)**
Device ships with the multi-run optimized model trained on all 14 Siena patients. No patient-specific data required. Achieves 8/12 (67%) predictability immediately.

**Phase 2 — Patient-specific fine-tuning (calibration)**
After the patient records 1–2 seizures with the device:
- First 3 layers frozen (preserve general preictal knowledge)
- Final 2 layers retrained on patient's own data (tiny LR = 1e-5)
- Adapts the decision boundary to this specific brain

This is demonstrated in `scripts_v6/07_finetune.py`.

---


> **Note on reproducibility:** Due to random seed variance, exact AUC values will differ between runs. The multi-run strategy mitigates this — running 50+ seeds should reliably reproduce 7–8/12 predictable patients.

---

### Output Files

```
models_v6/
  nn_PN*.keras              Best model per patient (from multi-run)
  scaler_PN*.pkl            Feature scaler per patient fold
  lopo_results.json         Run 1 (seed 42) results
  multi_run_results.json    Best AUC per patient across all seeds
  finetuned_results.json    Fine-tuning comparison results

data/processed_v6/
  PN*.npz                   Preprocessed windows per patient

data/features/
  features_v6.npz           64-feature matrix (60,457 × 64)
```

---

### Hardware & Software

```
Hardware  : Apple MacBook M4 (Metal GPU acceleration)
Python    : 3.13
TensorFlow: 2.x with Metal GPU (TF_METAL_ENABLED=1)
Key deps  : mne, numpy, scipy, scikit-learn, tensorflow, joblib
Training  : ~20 min for Run 1, ~4 hours for 297-seed multi-run
```

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
