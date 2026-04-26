"""
=============================================================
  Siena Scalp EEG — Feature Extraction v2.02
  04_extract_features.py

  Same 64 band power features as v2 — the features that
  proved best for cross-patient generalization.

  Features : 64
    Band powers    (40): delta, theta, alpha, beta, gamma x 8ch
    Band ratios    (16): theta/alpha, delta/beta x 8ch
    Spectral entropy(8): 1 per channel

  Normalization : z-score to patient interictal baseline
  Output        : data/features/features_v202.npz
=============================================================
"""

import numpy as np
from scipy import signal
from pathlib import Path

PROCESSED_DIR = Path("data/processed_v202")
FEATURES_DIR  = Path("data/features")
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


def extract_features(window: np.ndarray) -> np.ndarray:
    freqs, psd = signal.welch(window, fs=SFREQ, nperseg=512, axis=-1)
    eps = 1e-10

    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[:, b_idx] = np.log1p(psd[:, mask].mean(axis=1))

    theta_alpha = (band_powers[:, BAND_NAMES.index("theta")] /
                   (band_powers[:, BAND_NAMES.index("alpha")] + eps))
    delta_beta  = (band_powers[:, BAND_NAMES.index("delta")] /
                   (band_powers[:, BAND_NAMES.index("beta")]  + eps))

    total   = psd.sum(axis=1, keepdims=True) + eps
    pn      = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)

    return np.concatenate([
        band_powers.flatten(),
        theta_alpha,
        delta_beta,
        entropy,
    ]).astype(np.float32)


def normalize_to_baseline(X, y):
    mask = y == 0
    if mask.sum() < 10:
        return X
    mean = X[mask].mean(axis=0)
    std  = X[mask].std(axis=0)
    std[std < 1e-10] = 1.0
    return ((X - mean) / std).astype(np.float32)


print("\n" + "=" * 62)
print("  FEATURE EXTRACTION v2.02")
print("=" * 62)
print(f"  Band powers (40) + ratios (16) + entropy (8) = 64 features")
print(f"  Normalization: z-score to patient interictal baseline")
print("=" * 62 + "\n")

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print(f"No .npz files in {PROCESSED_DIR} -- run 03_preprocess.py first")
    exit(1)

all_X, all_y, all_patients = [], [], []

for npz_path in npz_files:
    patient   = npz_path.stem
    data      = np.load(str(npz_path))
    X_raw     = data["X"]
    y         = data["y"]
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"-- {patient}  ({len(X_raw)} windows | "
          f"pre: {n_pre} | inter: {n_inter})")

    X_feat = np.array([extract_features(X_raw[i])
                       for i in range(len(X_raw))],
                      dtype=np.float32)

    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    X_norm = normalize_to_baseline(X_feat, y)
    print(f"   mean: {X_norm.mean():.4f}  std: {X_norm.std():.4f}")

    all_X.append(X_norm)
    all_y.append(y)
    all_patients.extend([patient] * len(X_raw))

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
print(f"  Imbalance      : {int((y_all==0).sum())/int((y_all==1).sum()):.1f}:1")
print(f"  Mean           : {X_all.mean():.4f}")
print(f"  Std            : {X_all.std():.4f}")

out_path = FEATURES_DIR / "features_v202.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n  Saved -> {out_path}")
print(f"{'=' * 62}\n")