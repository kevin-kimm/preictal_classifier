"""
=============================================================
  Siena Scalp EEG — Band Power Feature Extraction v2
  04_extract_features.py

  Key improvement over v1:
    Features are normalized relative to each patients own
    interictal baseline. This removes between-patient EEG
    amplitude differences that prevent cross-patient learning.

    For each patient:
      1. Compute raw features for every window
      2. Compute mean + std of interictal windows only
      3. Z-score all windows: (x - interictal_mean) / interictal_std

    Result: features represent HOW MUCH each window deviates
    from that patient's normal brain state — not absolute values.

  Features per window (64 total):
    [0:40]  Band powers     8 ch × 5 bands  (log scale)
    [40:56] Band ratios     8 ch × 2 ratios (theta/alpha, delta/beta)
    [56:64] Spectral entropy 8 ch

  Output: data/features/features.npz
    X        : (n_windows, 64)  float32 — relative features
    y        : (n_windows,)     int8
    patients : (n_windows,)     str
=============================================================
"""

import numpy as np
from scipy import signal
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 250

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDS.keys())
N_CHANNELS  = 8
N_BANDS     = len(BANDS)


# Raw feature extraction
# Input : (8, 7500)
# Output: (64,)
def extract_features_raw(window: np.ndarray) -> np.ndarray:
    """Compute absolute features for one window."""
    freqs, psd = signal.welch(window, fs=SFREQ, nperseg=512, axis=-1)

    # Band powers 
    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[:, b_idx] = np.log1p(psd[:, mask].mean(axis=1))
    features = [band_powers.flatten()]

    # Band ratios 
    eps         = 1e-10
    theta_idx   = BAND_NAMES.index("theta")
    alpha_idx   = BAND_NAMES.index("alpha")
    delta_idx   = BAND_NAMES.index("delta")
    beta_idx    = BAND_NAMES.index("beta")
    theta_alpha = band_powers[:, theta_idx] / (band_powers[:, alpha_idx] + eps)
    delta_beta  = band_powers[:, delta_idx] / (band_powers[:, beta_idx]  + eps)
    features.append(theta_alpha)
    features.append(delta_beta)

    # Spectral entropy 
    total  = psd.sum(axis=1, keepdims=True) + eps
    pn     = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)
    features.append(entropy)

    return np.concatenate(features).astype(np.float32)


# Patient relative normalization
def normalize_to_baseline(X: np.ndarray,
                           y: np.ndarray) -> np.ndarray:
    """
    Z-score features relative to each patient's interictal baseline.

    Steps:
      1. Find all interictal windows (y == 0)
      2. Compute mean and std of those windows
      3. Z-score ALL windows (preictal + interictal) using
         the interictal mean/std

    Why interictal only for mean
      We want to measure deviation from NORMAL brain state.
      Including preictal windows in the baseline would
      dilute the signal we're trying to detect.
    """
    interictal_mask = y == 0
    n_inter = interictal_mask.sum()

    if n_inter < 10:
        print(f"    Only {n_inter} interictal windows — skipping normalization")
        return X

    baseline_mean = X[interictal_mask].mean(axis=0)
    baseline_std  = X[interictal_mask].std(axis=0)

    # Replace zero std with 1 to avoid division by zero
    baseline_std[baseline_std < 1e-10] = 1.0

    X_normalized = (X - baseline_mean) / baseline_std
    return X_normalized.astype(np.float32)


# Process all patients
print("\n" + "=" * 62)
print("  feature extraction v2 — patient relative normalization")
print("=" * 62)
print(f"  Bands    : {BAND_NAMES}")
print(f"  Features : {N_CHANNELS}ch × {N_BANDS} bands = {N_CHANNELS*N_BANDS}")
print(f"             {N_CHANNELS}ch × 2 ratios        = {N_CHANNELS*2}")
print(f"             {N_CHANNELS}ch × 1 entropy       = {N_CHANNELS}")
print(f"             Total = {N_CHANNELS*N_BANDS + N_CHANNELS*2 + N_CHANNELS}")
print(f"\n  Normalization: z score relative to interictal baseline")
print("=" * 62 + "\n")

all_X        = []
all_y        = []
all_patients = []

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print("No .npz files found in data/processed/")
    print("   Run 03_preprocess.py first.")
    exit(1)

for npz_path in npz_files:
    patient = npz_path.stem
    data    = np.load(str(npz_path))
    X_raw   = data["X"]  # (N, 8, 7500)
    y       = data["y"]  # (N,)

    n_windows = len(X_raw)
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"── {patient}  ({n_windows} windows | "
          f"pre: {n_pre} | inter: {n_inter})")

    # Step 1: extract raw features 
    X_feat = np.array([
        extract_features_raw(X_raw[i])
        for i in range(n_windows)
    ], dtype=np.float32)

    # Step 2: check
    if not np.isfinite(X_feat).all():
        n_bad = (~np.isfinite(X_feat)).sum()
        print(f"     {n_bad} non-finite values — clipping")
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    # Step 3: normalize relative to interictal baseline 
    X_norm = normalize_to_baseline(X_feat, y)

    # Print before/after stats for first patient 
    if patient == "PN00":
        print(f"   Raw    — mean: {X_feat.mean():>10.2f}  "
              f"std: {X_feat.std():>12.2f}  "
              f"max: {X_feat.max():>15.2f}")
        print(f"   Normed — mean: {X_norm.mean():>10.4f}  "
              f"std: {X_norm.std():>12.4f}  "
              f"max: {X_norm.max():>15.4f}")

    all_X.append(X_norm)
    all_y.append(y)
    all_patients.extend([patient] * n_windows)

# Concatenate all patients 
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
p_all = np.array(all_patients)

n_pre   = int((y_all == 1).sum())
n_inter = int((y_all == 0).sum())

print(f"\n{'=' * 62}")
print(f"  COMPLETE")
print(f"{'=' * 62}")
print(f"  Total windows  : {len(y_all):,}")
print(f"  Feature shape  : {X_all.shape}")
print(f"  Preictal       : {n_pre:,}")
print(f"  Interictal     : {n_inter:,}")
print(f"  Imbalance      : {n_inter/n_pre:.1f}:1")
print(f"\n  Feature stats (all patients combined):")
print(f"    Mean  : {X_all.mean():.4f}")
print(f"    Std   : {X_all.std():.4f}")
print(f"    Min   : {X_all.min():.4f}")
print(f"    Max   : {X_all.max():.4f}")

# Save
out_path = FEATURES_DIR / "features.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n   Saved → {out_path}")
print(f"{'=' * 62}\n")