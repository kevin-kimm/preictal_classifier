"""
=============================================================
  Siena Scalp EEG — Feature Extraction v6
  04_extract_features.py

  64 band power features — proven best for cross-patient
  generalization on 14 patients.

  Features:
    Band powers    (40): delta,theta,alpha,beta,gamma x 8ch
    Band ratios    (16): theta/alpha, delta/beta x 8ch
    Spectral entropy(8): 1 per channel

  Normalization: z-score relative to patient interictal baseline
  Output: data/features/features_v6.npz
=============================================================
"""


import numpy as np
from scipy import signal
from pathlib import Path


# File paths
# processed windows from 03_preprocess.py
PROCESSED_DIR = Path("data/processed_v6")

# Output: feature matrix ready for model training
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# Configuration 
SFREQ      = 250   # sampling frequency (Hz)
N_CHANNELS = 8     # F7,T3,T5,C3,F8,T4,T6,C4
N_BANDS    = 5     # delta, theta, alpha, beta, gamma

# Frequency bands and their ranges (Hz)
BANDS = {
    "delta": (0.5,  4.0),   # deep sleep, seizure activity
    "theta": (4.0,  8.0),   # drowsiness, preictal changes
    "alpha": (8.0,  13.0),  # relaxed wakefulness
    "beta":  (13.0, 30.0),  # active thinking, alertness
    "gamma": (30.0, 40.0),  # intense focus, seizure onset
}
BAND_NAMES = list(BANDS.keys())  # ["delta","theta","alpha","beta","gamma"]



# FEATURE EXTRACTION FUNCTION
# Converts one 30-second EEG window (8 channels × 7500 samples)
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extracts 64 features from one 30-second EEG window.

    window : numpy array, shape (8_channels, 7500_samples)
    returns: numpy array, shape (64,)

    The 64 features are:
      40 band powers    (8 channels × 5 frequency bands)
      16 band ratios    (8 channels × 2 ratios)
       8 spectral entropy (1 per channel)
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
    # psd   : power at each frequency, shape (8_channels, n_freqs)
    freqs, psd = signal.welch(window, fs=SFREQ,
                               nperseg=512, axis=-1)
    eps = 1e-10  # small constant to prevent log(0) and division by zero

    # STEP 2: Band Powers (40 features)
    # For each channel and each frequency band:
    # Find all frequency bins within the band range
    # Average the PSD values in that range
    # Apply log1p (log(1+x)) to compress the scale
    #   (brain power spans many orders of magnitude)
    band_powers = np.zeros((N_CHANNELS, N_BANDS))
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        # Boolean mask: True for frequencies within this band
        mask = (freqs >= lo) & (freqs < hi)
        # Mean power in band, log-compressed, for all 8 channels
        band_powers[:, b_idx] = np.log1p(
            psd[:, mask].mean(axis=1))

    # band_powers shape: (8_channels, 5_bands)
    # When flattened: [F7_delta, F7_theta, ..., C4_gamma] = 40 features

    # STEP 3: Band Ratios (16 features) 
    # Ratios are more informative than raw powers because
    # they normalize out individual differences in brain amplitude
    #
    # Theta/Alpha ratio:
    #   Normal: alpha dominant → ratio ~0.5-1.0
    #   Preictal: theta rises, alpha drops → ratio > 1.5
    #   One of the most validated preictal biomarkers in literature
    #
    # Delta/Beta ratio:
    #   Normal: delta low, beta moderate → ratio ~0.3
    #   Preictal: delta increases, consciousness may decrease
    theta_alpha = (
        band_powers[:, BAND_NAMES.index("theta")] /
        (band_powers[:, BAND_NAMES.index("alpha")] + eps))
    # shape: (8_channels,) — one ratio per channel = 8 features

    delta_beta = (
        band_powers[:, BAND_NAMES.index("delta")] /
        (band_powers[:, BAND_NAMES.index("beta")]  + eps))
    # shape: (8_channels,) — one ratio per channel = 8 features

    # STEP 4: Spectral Entropy (8 features) 
    # Shannon entropy applied to the power spectrum
    #
    # Intuition:
    #   High entropy = power spread across many frequencies
    #              = complex, unpredictable = NORMAL brain
    #   Low entropy  = power concentrated in few frequencies
    #              = rhythmic, synchronized = PREICTAL/ICTAL
    #
    # Pre-seizure brains start synchronizing toward one frequency
    # This reduces entropy — detectable before the seizure starts
    #
    # Formula: H = -sum(p(f) × log(p(f)))
    # where p(f) = PSD(f) / sum(PSD)  (normalized to probability)
    total   = psd.sum(axis=1, keepdims=True) + eps
    pn      = np.clip(psd / total, eps, 1)   # normalize to [0,1]
    entropy = -(pn * np.log(pn)).sum(axis=1) # Shannon entropy
    # shape: (8_channels,) — one entropy value per channel = 8 features

    # STEP 5: Concatenate all features 
    # Stack into single 64-element vector:
    # [40 band powers | 8 theta/alpha ratios | 8 delta/beta ratios | 8 entropies]
    return np.concatenate([
        band_powers.flatten(),  # 40 features
        theta_alpha,            #  8 features
        delta_beta,             #  8 features
        entropy,                #  8 features
    ]).astype(np.float32)       # = 64 features total



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
    # Use only interictal windows to compute the baseline
    # (preictal windows are already different — don't include them)
    mask = y == 0
    if mask.sum() < 10:
        return X  # not enough interictal data to normalize
    mean = X[mask].mean(axis=0)  # mean per feature
    std  = X[mask].std(axis=0)   # std per feature
    std[std < 1e-10] = 1.0       # prevent division by zero
    return ((X - mean) / std).astype(np.float32)


# MAIN PROCESSING LOOP
print("\n" + "=" * 62)
print("  FEATURE EXTRACTION v6")
print("=" * 62)
print(f"  Band powers (40) + ratios (16) + entropy (8) = 64")
print(f"  Normalization: z-score to patient interictal baseline")
print("=" * 62 + "\n")

# Find all processed patient files
npz_files = sorted(PROCESSED_DIR.glob("*.npz"))
if not npz_files:
    print(f"No .npz files found in {PROCESSED_DIR}")
    print("Run 03_preprocess.py first.")
    exit(1)

# Lists to accumulate data across all patients
all_X, all_y, all_patients = [], [], []

for npz_path in npz_files:
    patient   = npz_path.stem          # e.g. "PN07"
    data      = np.load(str(npz_path))
    X_raw     = data["X"]              # shape: (n_windows, 8, 7500)
    y         = data["y"]              # shape: (n_windows,)
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"-- {patient}  ({len(X_raw)} windows | "
          f"pre:{n_pre} | inter:{n_inter})")

    # Extract 64 features from each window
    # List comprehension processes one window at a time
    X_feat = np.array([extract_features(X_raw[i])
                       for i in range(len(X_raw))],
                      dtype=np.float32)
    # X_feat shape: (n_windows, 64)

    # Replace any NaN/Inf values with 0 (safety check)
    if not np.isfinite(X_feat).all():
        X_feat = np.nan_to_num(X_feat, nan=0.0,
                                posinf=0.0, neginf=0.0)

    # Normalize relative to this patient's interictal baseline
    X_norm = normalize_to_baseline(X_feat, y)
    print(f"   mean:{X_norm.mean():.4f}  std:{X_norm.std():.4f}")
    # After normalization: mean≈0, std≈1 for interictal windows
    # Preictal windows will deviate from this baseline

    all_X.append(X_norm)
    all_y.append(y)
    # Track which patient each window belongs to
    # Used in 05_train_model.py for LOPO splitting
    all_patients.extend([patient] * len(X_raw))

# Stack all patients into one big matrix
X_all = np.concatenate(all_X, axis=0)   # shape: (total_windows, 64)
y_all = np.concatenate(all_y, axis=0)   # shape: (total_windows,)
p_all = np.array(all_patients)           # shape: (total_windows,)

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

# Save as compressed numpy file
# .npz = multiple arrays in one file (like a zip)
# X        : feature matrix (n_windows, 64)
# y        : labels (n_windows,)
# patients : patient ID per window (n_windows,)
out_path = FEATURES_DIR / "features_v6.npz"
np.savez_compressed(str(out_path), X=X_all,
                    y=y_all, patients=p_all)
print(f"\n  Saved -> {out_path}")
print(f"{'=' * 62}\n")