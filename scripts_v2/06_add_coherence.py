"""
=============================================================
  Siena Scalp EEG — Coherence Feature Extraction
  06_add_coherence.py

  Adds inter-channel coherence features to existing band
  power features.

  Coherence measures how synchronized two channels are at
  each frequency band. Pre-seizure brains show increasing
  synchronization between temporal regions — one of the
  most well-established seizure biomarkers.

  New features per window:
    Band power features  :  64  (from 04_extract_features.py)
    Coherence features   : 140  (28 pairs × 5 bands)
    Total                : 204

  28 channel pairs from 8 channels:
    C(8,2) = 8! / (2! × 6!) = 28 pairs

  Output: data/features/features_coherence.npz
=============================================================
"""

import numpy as np
from scipy import signal
from itertools import combinations
from pathlib import Path

PROCESSED_DIR  = Path("data/processed")
FEATURES_DIR   = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SFREQ      = 250
N_CHANNELS = 8
N_BANDS    = 5

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDS.keys())

# All unique channel pairs
CHANNEL_PAIRS = list(combinations(range(N_CHANNELS), 2))
N_PAIRS       = len(CHANNEL_PAIRS)  # 28


# Band power featured same as 04_extract_features.py
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

    total  = psd.sum(axis=1, keepdims=True) + eps
    pn     = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)

    return np.concatenate([
        band_powers.flatten(),
        theta_alpha,
        delta_beta,
        entropy
    ]).astype(np.float32)


# Coherence features
def extract_coherence(window: np.ndarray) -> np.ndarray:
    """
    Compute magnitude-squared coherence for every channel pair
    in each frequency band.

    Coherence ranges from 0 (no synchronization) to 1 (perfect
    synchronization). We use magnitude-squared coherence (MSC)
    which is the standard in EEG research.

    For each pair (i,j) and each band:
      MSC = |Pxy|^2 / (Pxx * Pyy)
      where Pxy = cross-spectral density
            Pxx, Pyy = auto-spectral densities

    Returns: (N_PAIRS × N_BANDS,) = (28 × 5,) = 140 features
    """
    coherence_features = np.zeros((N_PAIRS, N_BANDS),
                                   dtype=np.float32)

    for pair_idx, (ch_i, ch_j) in enumerate(CHANNEL_PAIRS):
        # Compute coherence between channel i and channel j
        freqs, coh = signal.coherence(
            window[ch_i], window[ch_j],
            fs=SFREQ,
            nperseg=512
        )

        # Average coherence within each frequency band
        for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() > 0:
                coherence_features[pair_idx, b_idx] = coh[mask].mean()

    return coherence_features.flatten()


# Combined feature extraction 
def extract_all_features(window: np.ndarray) -> np.ndarray:
    """Extract band powers (64) + coherence (140) = 204 features."""
    bp  = extract_band_powers(window)   # 64
    coh = extract_coherence(window)     # 140
    return np.concatenate([bp, coh]).astype(np.float32)


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
n_bp_features  = N_CHANNELS * N_BANDS + N_CHANNELS * 2 + N_CHANNELS
n_coh_features = N_PAIRS * N_BANDS
n_total        = n_bp_features + n_coh_features

print("\n" + "=" * 62)
print("  COHERENCE FEATURE EXTRACTION")
print("=" * 62)
print(f"  Band power features : {n_bp_features}")
print(f"  Coherence features  : {N_PAIRS} pairs × {N_BANDS} bands = {n_coh_features}")
print(f"  Total features      : {n_total}")
print(f"  Channel pairs       : {N_PAIRS}")
print(f"\n  Channel pair examples:")
ch_names = ["T3", "T5", "O1", "O2", "T6", "T4", "F7", "F8"]
for i, (a, b) in enumerate(CHANNEL_PAIRS[:6]):
    print(f"    {ch_names[a]}–{ch_names[b]}")
print(f"    ... and {N_PAIRS - 6} more pairs")
print("=" * 62 + "\n")

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print("No .npz files found — run 03_preprocess.py first")
    exit(1)

all_X        = []
all_y        = []
all_patients = []

for npz_path in npz_files:
    patient   = npz_path.stem
    data      = np.load(str(npz_path))
    X_raw     = data["X"]   # (N, 8, 7500)
    y         = data["y"]   # (N,)
    n_windows = len(X_raw)
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"── {patient}  ({n_windows} windows | "
          f"pre: {n_pre} | inter: {n_inter})")

    # Extract features — coherence is slower so print progress
    X_feat = []
    for i in range(n_windows):
        X_feat.append(extract_all_features(X_raw[i]))
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{n_windows} windows processed...")

    X_feat = np.array(X_feat, dtype=np.float32)

    # Sanity check
    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    # Normalize relative to interictal baseline
    X_norm = normalize_to_baseline(X_feat, y)

    print(f"   Features: {X_feat.shape} → normalized "
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

out_path = FEATURES_DIR / "features_coherence.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n  Saved → {out_path}")
print(f"{'=' * 62}\n")
ß
