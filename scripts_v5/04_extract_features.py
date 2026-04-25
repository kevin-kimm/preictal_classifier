"""
=============================================================
  Siena Scalp EEG — Feature Extraction v5
  04_extract_features.py

  Best generalized detector:
    Band powers + ratios + entropy  : 64 features
    Asymmetry features (L-R diff)  :  5 features
    Total                          : 69 features

  Asymmetry features:
    F7-F8 (frontal)
    T3-T4 (temporal)
    T5-T6 (posterior temporal)
    C3-C4 (central) — proxy hemispheric difference
    (T3+T5)-(T4+T6) (overall left-right temporal)

  Channels : F7, T3, T5, C3, F8, T4, T6, C4
  Output   : data/features/features_v5.npz
=============================================================
"""

import numpy as np
from scipy import signal
from pathlib import Path

PROCESSED_DIR = Path("data/processed_v5")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SFREQ      = 250
N_CHANNELS = 8
# Channel order: F7=0, T3=1, T5=2, C3=3, F8=4, T4=5, T6=6, C4=7
CH_NAMES   = ["F7", "T3", "T5", "C3", "F8", "T4", "T6", "C4"]

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDS.keys())
N_BANDS    = len(BANDS)

# Asymmetry pairs (left_idx, right_idx, name)
ASYM_PAIRS = [
    (0, 4, "F7-F8"),          # frontal
    (1, 5, "T3-T4"),          # temporal
    (2, 6, "T5-T6"),          # posterior temporal
    (3, 7, "C3-C4"),          # central
]


# BAND POWERS (64 features)
def extract_band_powers(window: np.ndarray) -> np.ndarray:
    freqs, psd = signal.welch(window, fs=SFREQ, nperseg=512, axis=-1)
    eps = 1e-10

    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[:, b_idx] = np.log1p(psd[:, mask].mean(axis=1))

    theta_idx = BAND_NAMES.index("theta")
    alpha_idx = BAND_NAMES.index("alpha")
    delta_idx = BAND_NAMES.index("delta")
    beta_idx  = BAND_NAMES.index("beta")

    theta_alpha = band_powers[:, theta_idx] / \
                  (band_powers[:, alpha_idx] + eps)
    delta_beta  = band_powers[:, delta_idx]  / \
                  (band_powers[:, beta_idx]  + eps)

    total   = psd.sum(axis=1, keepdims=True) + eps
    pn      = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)

    return np.concatenate([
        band_powers.flatten(),   # 40
        theta_alpha,             #  8
        delta_beta,              #  8
        entropy,                 #  8
    ]).astype(np.float32)        # = 64



# ASYMMETRY FEATURES (5 features)
def extract_asymmetry(window: np.ndarray,
                       band_powers_flat: np.ndarray) -> np.ndarray:
    """
    Left-right power asymmetry per channel pair.

    Seizures typically start on one side and spread to the other.
    The asymmetry between hemispheres changes measurably in the
    preictal period as the affected side begins to synchronize.

    We use total band power (sum across all bands) per channel
    and compute normalized asymmetry:
      asym = (left - right) / (left + right + eps)

    Range: -1 (all power on right) to +1 (all power on left)
    Normal brain: near 0 (balanced)
    Preictal: shifts toward the seizure-onset hemisphere

    Returns 5 features:
      F7-F8, T3-T4, T5-T6, C3-C4, (T3+T5)-(T4+T6) overall
    """
    # Reshape band powers back to (8 channels, 5 bands)
    bp = band_powers_flat[:N_CHANNELS * N_BANDS].reshape(
        N_CHANNELS, N_BANDS)

    # Total power per channel (sum across bands)
    total_power = bp.sum(axis=1)  # (8,)

    eps = 1e-10
    asym = np.zeros(5, dtype=np.float32)

    for i, (l_idx, r_idx, _) in enumerate(ASYM_PAIRS):
        l = total_power[l_idx]
        r = total_power[r_idx]
        asym[i] = (l - r) / (l + r + eps)

    # Overall left temporal vs right temporal
    left_total  = total_power[1] + total_power[2]   # T3 + T5
    right_total = total_power[5] + total_power[6]   # T4 + T6
    asym[4]     = (left_total - right_total) / \
                  (left_total + right_total + eps)

    return asym


# COMBINED FEATURES (69 total)
def extract_all_features(window: np.ndarray) -> np.ndarray:
    bp   = extract_band_powers(window)          # 64
    asym = extract_asymmetry(window, bp)        #  5
    return np.concatenate([bp, asym]).astype(np.float32)  # 69


# PATIENT RELATIVE NORMALIZATION
def normalize_to_baseline(X: np.ndarray,
                           y: np.ndarray) -> np.ndarray:
    interictal_mask = y == 0
    if interictal_mask.sum() < 10:
        return X
    baseline_mean = X[interictal_mask].mean(axis=0)
    baseline_std  = X[interictal_mask].std(axis=0)
    baseline_std[baseline_std < 1e-10] = 1.0
    return ((X - baseline_mean) / baseline_std).astype(np.float32)


# MAIN
print("\n" + "=" * 62)
print("  FEATURE EXTRACTION v5")
print("=" * 62)
print(f"  Channels  : {CH_NAMES}")
print(f"  Band powers + ratios + entropy : 64 features")
print(f"  Asymmetry pairs (L-R)          :  5 features")
for _, (l, r, name) in enumerate(ASYM_PAIRS):
    print(f"    {name}")
print(f"    (T3+T5)-(T4+T6) overall")
print(f"  Total features                 : 69")
print(f"  Normalization : z-score to patient interictal baseline")
print("=" * 62 + "\n")

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print(f"No .npz files in {PROCESSED_DIR}")
    print("Run 03_preprocess.py first.")
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

    print(f"-- {patient}  ({n_windows} windows | "
          f"pre: {n_pre} | inter: {n_inter})")

    X_feat = []
    for i in range(n_windows):
        X_feat.append(extract_all_features(X_raw[i]))
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{n_windows} windows...")

    X_feat = np.array(X_feat, dtype=np.float32)

    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    X_norm = normalize_to_baseline(X_feat, y)
    print(f"   mean: {X_norm.mean():.4f}  std: {X_norm.std():.4f}")

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
print(f"  Imbalance      : {int((y_all==0).sum())/int((y_all==1).sum()):.1f}:1")
print(f"  Mean           : {X_all.mean():.4f}")
print(f"  Std            : {X_all.std():.4f}")

out_path = FEATURES_DIR / "features_v5.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n  Saved -> {out_path}")
print(f"{'=' * 62}\n")