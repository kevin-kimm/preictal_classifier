"""
=============================================================
  Siena Scalp EEG — Correlation Feature Extraction
  07_add_correlation.py

  Adds time-domain correlation features to band power features.
  These are fundamentally different from coherence (frequency
  domain) and capture different aspects of brain synchronization.

  New features per window:
    Band powers (64) + Correlation (28) + Cross-corr (56) = 148

  Correlation (28 features):
    Pearson correlation between every channel pair.
    Measures overall linear synchrony in the time domain.
    Pre-seizure: temporal channels increasingly correlate.

  Cross-correlation (56 features):
    Peak cross-correlation value + lag for every pair.
    Captures propagation delay as seizures spread across brain.
    Pre-seizure: propagation patterns change measurably.

  Output: data/features/features_correlation.npz
=============================================================
"""

import numpy as np
from scipy import signal
from itertools import combinations
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SFREQ      = 250
N_CHANNELS = 8
N_BANDS    = 5
MAX_LAG_MS = 500   # max cross-correlation lag in milliseconds
MAX_LAG    = int(MAX_LAG_MS * SFREQ / 1000)  # in samples = 125

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES    = list(BANDS.keys())
CHANNEL_PAIRS = list(combinations(range(N_CHANNELS), 2))
N_PAIRS       = len(CHANNEL_PAIRS)  # 28

CH_NAMES = ["T3", "T5", "O1", "O2", "T6", "T4", "F7", "F8"]


# Band power features (unchanged from 04_extract_features.py)
def extract_band_powers(window: np.ndarray) -> np.ndarray:
    freqs, psd = signal.welch(window, fs=SFREQ, nperseg=512, axis=-1)

    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[:, b_idx] = np.log1p(psd[:, mask].mean(axis=1))

    eps        = 1e-10
    theta_idx  = BAND_NAMES.index("theta")
    alpha_idx  = BAND_NAMES.index("alpha")
    delta_idx  = BAND_NAMES.index("delta")
    beta_idx   = BAND_NAMES.index("beta")

    theta_alpha = band_powers[:, theta_idx] / (band_powers[:, alpha_idx] + eps)
    delta_beta  = band_powers[:, delta_idx] / (band_powers[:, beta_idx]  + eps)

    total   = psd.sum(axis=1, keepdims=True) + eps
    pn      = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)

    return np.concatenate([
        band_powers.flatten(),
        theta_alpha,
        delta_beta,
        entropy
    ]).astype(np.float32)


# Pearson correlation 28 features 
def extract_correlation(window: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between every channel pair.

    We z-score each channel first so correlation is not
    affected by amplitude differences between channels.

    Returns: (28,) — one value per pair, range [-1, 1]
    """
    # Z-score each channel
    mean = window.mean(axis=1, keepdims=True)
    std  = window.std(axis=1,  keepdims=True)
    std[std < 1e-10] = 1.0
    z = (window - mean) / std  # (8, 7500)

    corr_features = np.zeros(N_PAIRS, dtype=np.float32)
    for pair_idx, (i, j) in enumerate(CHANNEL_PAIRS):
        # Pearson correlation = mean of element-wise product of z-scores
        corr_features[pair_idx] = np.mean(z[i] * z[j])

    return corr_features



# Cross correlation 56 features = 28 pairs × 2 values each
def extract_cross_correlation(window: np.ndarray) -> np.ndarray:
    """
    For each channel pair compute:
      1. Peak cross-correlation value (how strongly do they sync?)
      2. Lag at peak (how many ms does one lead the other?)

    We limit the lag to ±500ms — beyond that it's unlikely to
    reflect true neural propagation in epilepsy.

    Returns: (56,) — peak_corr and lag for each of 28 pairs
    """
    # Z-score each channel
    mean = window.mean(axis=1, keepdims=True)
    std  = window.std(axis=1,  keepdims=True)
    std[std < 1e-10] = 1.0
    z = (window - mean) / std  # (8, 7500)

    n_samples = window.shape[1]
    xcorr_features = np.zeros(N_PAIRS * 2, dtype=np.float32)

    for pair_idx, (i, j) in enumerate(CHANNEL_PAIRS):
        # Full cross correlation
        xcorr = np.correlate(z[i], z[j], mode="full")
        xcorr = xcorr / n_samples  # normalize

        # Only look at lags within ±MAX_LAG samples
        center   = len(xcorr) // 2
        lag_range = xcorr[center - MAX_LAG: center + MAX_LAG + 1]

        # Peak value and its lag
        peak_idx  = np.argmax(np.abs(lag_range))
        peak_val  = lag_range[peak_idx]
        peak_lag  = (peak_idx - MAX_LAG) / SFREQ * 1000  # convert to ms

        xcorr_features[pair_idx * 2]     = float(peak_val)
        xcorr_features[pair_idx * 2 + 1] = float(peak_lag)

    return xcorr_features


# Combined feature extraction
def extract_all_features(window: np.ndarray) -> np.ndarray:
    """
    Extract all features: band powers (64) + correlation (28)
    + cross-correlation (56) = 148 total.
    """
    bp    = extract_band_powers(window)       # 64
    corr  = extract_correlation(window)       # 28
    xcorr = extract_cross_correlation(window) # 56
    return np.concatenate([bp, corr, xcorr]).astype(np.float32)


# Patient relative normalization
def normalize_to_baseline(X: np.ndarray,
                           y: np.ndarray) -> np.ndarray:
    interictal_mask = y == 0
    if interictal_mask.sum() < 10:
        return X
    baseline_mean = X[interictal_mask].mean(axis=0)
    baseline_std  = X[interictal_mask].std(axis=0)
    baseline_std[baseline_std < 1e-10] = 1.0
    return ((X - baseline_mean) / baseline_std).astype(np.float32)


# Main
n_bp    = N_CHANNELS * N_BANDS + N_CHANNELS * 2 + N_CHANNELS
n_corr  = N_PAIRS
n_xcorr = N_PAIRS * 2
n_total = n_bp + n_corr + n_xcorr

print("\n" + "=" * 62)
print("  CORRELATION FEATURE EXTRACTION")
print("=" * 62)
print(f"  Band powers     : {n_bp} features")
print(f"  Correlation     : {n_corr} features ({N_PAIRS} pairs)")
print(f"  Cross-corr      : {n_xcorr} features ({N_PAIRS} pairs × 2)")
print(f"  Total           : {n_total} features")
print(f"  Max lag         : ±{MAX_LAG_MS}ms ({MAX_LAG} samples)")
print(f"\n  Example channel pairs:")
for i, j in CHANNEL_PAIRS[:5]:
    print(f"    {CH_NAMES[i]}–{CH_NAMES[j]}")
print(f"    ... and {N_PAIRS - 5} more")
print("=" * 62 + "\n")

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print("No .npz files — run 03_preprocess.py first")
    exit(1)

all_X        = []
all_y        = []
all_patients = []

for npz_path in npz_files:
    patient   = npz_path.stem
    data      = np.load(str(npz_path))
    X_raw     = data["X"]
    y         = data["y"]
    n_windows = len(X_raw)
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"── {patient}  ({n_windows} windows | "
          f"pre: {n_pre} | inter: {n_inter})")

    X_feat = []
    for i in range(n_windows):
        X_feat.append(extract_all_features(X_raw[i]))
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{n_windows} windows...")

    X_feat = np.array(X_feat, dtype=np.float32)

    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    X_norm = normalize_to_baseline(X_feat, y)
    print(f"   Features: {X_feat.shape} → "
          f"mean: {X_norm.mean():.3f}  std: {X_norm.std():.3f}")

    all_X.append(X_norm)
    all_y.append(y)
    all_patients.extend([patient] * n_windows)

X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
p_all = np.array(all_patients)

print(f"\n{'=' * 62}")
print(f"  COMPLETE")
print(f"{'=' * 62}")
print(f"  Total windows  : {len(y_all):,}")
print(f"  Feature shape  : {X_all.shape}")
print(f"  Preictal       : {int((y_all==1).sum()):,}")
print(f"  Interictal     : {int((y_all==0).sum()):,}")
print(f"  Mean           : {X_all.mean():.4f}")
print(f"  Std            : {X_all.std():.4f}")

out_path = FEATURES_DIR / "features_correlation.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n  Saved: {out_path}")
print(f"{'=' * 62}\n")