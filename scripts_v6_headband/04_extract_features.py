"""
=============================================================
  Siena Scalp EEG — Feature Extraction v6 Headband
  04_extract_features.py

  72 band power features — 9 channels × 8 feature types
  Same proven feature set as v6 but for 9-channel hybrid montage.

  Features:
    Band powers    (45): delta,theta,alpha,beta,gamma x 9ch
    Band ratios    (18): theta/alpha, delta/beta x 9ch
    Spectral entropy(9): 1 per channel
    Total          (72)

  Normalization: z-score relative to patient interictal baseline
  Output: data/features/features_v6_headband.npz
=============================================================
"""

# scipy   : signal processing — Welch PSD is here
import numpy as np
from scipy import signal
from pathlib import Path


# File paths 
PROCESSED_DIR = Path("data/processed_v6_headband")

# Output: feature matrix ready for model training
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# Configuration 
SFREQ      = 250   # sampling frequency (Hz)
N_CHANNELS = 9     # F7,T3,T5,O1,Pz,O2,T6,T4,F8
N_BANDS    = 5     # delta, theta, alpha, beta, gamma

# Frequency bands and their ranges (Hz)
# These are the clinically established EEG frequency bands
BANDS = {
    "delta": (0.5,  4.0),   # deep sleep, seizure activity
    "theta": (4.0,  8.0),   # drowsiness, preictal changes
    "alpha": (8.0,  13.0),  # relaxed wakefulness
    "beta":  (13.0, 30.0),  # active thinking, alertness
    "gamma": (30.0, 40.0),  # intense focus, seizure onset
}
BAND_NAMES = list(BANDS.keys())


# FEATURE EXTRACTION FUNCTION
# Converts one 30-second EEG window (9 channels × 7500 samples)
# into 72 meaningful numbers
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts 72 features from one 30-second EEG window.

    window : numpy array, shape (9_channels, 7500_samples)
    returns: numpy array, shape (72,)

    The 72 features are:
      45 band powers    (9 channels × 5 frequency bands)
      18 band ratios    (9 channels × 2 ratios)
       9 spectral entropy (1 per channel)
    """

    # STEP 1: Power Spectral Density via Welch's method 
    # Welch's method:
    #   1. Divide window into overlapping sub-windows (512 samples)
    #   2. Apply Hann window function to each sub-window
    #      (reduces spectral leakage at edges)
    #   3. Compute FFT on each sub-window
    #   4. Average the power spectra → stable PSD estimate
    #
    # freqs : array of frequency values [0, 0.49, 0.98, ... 125Hz]
    # psd   : power at each frequency, shape (9_channels, n_freqs)
    freqs, psd = signal.welch(window, fs=SFREQ,
                               nperseg=512, axis=-1)
    eps = 1e-10  # prevent log(0) and division by zero

    # STEP 2: Band Powers (45 features)
    # For each channel and each frequency band:
    # Find all frequency bins within the band range
    # Average the PSD values in that range
    # Apply log1p to compress the scale
    # (brain power spans many orders of magnitude)
    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[:, b_idx] = np.log1p(
            psd[:, mask].mean(axis=1))

    # STEP 3: Band Ratios (18 features)
    # Ratios normalize out individual brain amplitude differences
    #
    # Theta/Alpha ratio:
    #   Normal: alpha dominant → ratio ~0.5-1.0
    #   Preictal: theta rises, alpha drops → ratio > 1.5
    #   Most validated preictal biomarker in EEG literature
    #
    # Delta/Beta ratio:
    #   Normal: delta low → ratio ~0.3
    #   Preictal: delta increases as consciousness changes
    theta_alpha = (
        band_powers[:, BAND_NAMES.index("theta")] /
        (band_powers[:, BAND_NAMES.index("alpha")] + eps))
    # shape: (9_channels,) = 9 features

    delta_beta = (
        band_powers[:, BAND_NAMES.index("delta")] /
        (band_powers[:, BAND_NAMES.index("beta")]  + eps))
    # shape: (9_channels,) = 9 features

    # STEP 4: Spectral Entropy (9 features) 
    # Shannon entropy applied to the power spectrum
    #
    # High entropy = power spread across many frequencies
    #              = complex, unpredictable = NORMAL brain
    # Low entropy  = power concentrated in few frequencies
    #              = rhythmic, synchronized = PREICTAL
    #
    # Pre-seizure brains synchronize toward one frequency
    # reducing entropy — detectable before seizure starts
    #
    # Formula: H = -sum(p(f) × log(p(f)))
    # where p(f) = PSD(f) / sum(PSD)
    total   = psd.sum(axis=1, keepdims=True) + eps
    pn      = np.clip(psd / total, eps, 1)
    entropy = -(pn * np.log(pn)).sum(axis=1)
    # shape: (9_channels,) = 9 features

    # STEP 5: Concatenate all features 
    # [45 band powers | 9 theta/alpha | 9 delta/beta | 9 entropy]
    return np.concatenate([
        band_powers.flatten(),  # 45 features
        theta_alpha,            #  9 features
        delta_beta,             #  9 features
        entropy,                #  9 features
    ]).astype(np.float32)       # = 72 features total


# PATIENT-RELATIVE NORMALIZATION
def normalize_to_baseline(X, y):
    """
    Z-score normalization relative to each patient's
    own interictal baseline.

    Instead of asking "is theta power high?"
    We ask "is theta power HIGH FOR THIS PERSON?"

    This is critical because every brain is different.
    PN07's normal theta might be PN03's high theta.
    Without this, the model confuses natural variation
    between people with actual preictal changes.

    Formula per feature:
      normalized = (value - patient_interictal_mean)
                   / patient_interictal_std

    After normalization:
      0.0  = exactly this patient's average interictal level
      +2.0 = 2 standard deviations above their normal
      -1.0 = 1 standard deviation below their normal
    """
    mask = y == 0
    if mask.sum() < 10:
        return X
    mean = X[mask].mean(axis=0)
    std  = X[mask].std(axis=0)
    std[std < 1e-10] = 1.0
    return ((X - mean) / std).astype(np.float32)


# MAIN PROCESSING LOOP
print("\n" + "=" * 62)
print("  FEATURE EXTRACTION v6 HEADBAND")
print("=" * 62)
print(f"  Channels  : 9 (F7,T3,T5,O1,Pz,O2,T6,T4,F8)")
print(f"  Band powers (45) + ratios (18) + entropy (9) = 72")
print(f"  Normalization: z-score to patient interictal baseline")
print("=" * 62 + "\n")

npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print(f"No .npz files found in {PROCESSED_DIR}")
    print("Run 03_preprocess.py first.")
    exit(1)

all_X, all_y, all_patients = [], [], []

for npz_path in npz_files:
    patient   = npz_path.stem
    data      = np.load(str(npz_path))
    X_raw     = data["X"]              # shape: (n_windows, 9, 7500)
    y         = data["y"]
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"-- {patient}  ({len(X_raw)} windows | "
          f"pre:{n_pre} | inter:{n_inter})")

    # Extract 72 features from each window
    X_feat = np.array([extract_features(X_raw[i])
                       for i in range(len(X_raw))],
                      dtype=np.float32)

    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    X_norm = normalize_to_baseline(X_feat, y)
    print(f"   mean:{X_norm.mean():.4f}  std:{X_norm.std():.4f}")

    all_X.append(X_norm)
    all_y.append(y)
    all_patients.extend([patient] * len(X_raw))

X_all = np.concatenate(all_X, axis=0)   # shape: (total_windows, 72)
y_all = np.concatenate(all_y, axis=0)
p_all = np.array(all_patients)

print(f"\n{'=' * 62}")
print(f"  COMPLETE")
print(f"{'=' * 62}")
print(f"  Total windows : {len(y_all):,}")
print(f"  Feature shape : {X_all.shape}")
print(f"  Preictal      : {int((y_all==1).sum()):,}")
print(f"  Interictal    : {int((y_all==0).sum()):,}")
print(f"  Imbalance     : {int((y_all==0).sum())/int((y_all==1).sum()):.1f}:1")
print(f"  Mean          : {X_all.mean():.4f}")
print(f"  Std           : {X_all.std():.4f}")

out_path = FEATURES_DIR / "features_v6_headband.npz"
np.savez_compressed(str(out_path), X=X_all,
                    y=y_all, patients=p_all)
print(f"\n  Saved -> {out_path}")
print(f"{'=' * 62}\n")