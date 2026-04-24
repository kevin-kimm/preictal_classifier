"""
=============================================================
  Siena Scalp EEG — Feature Extraction v3
  04_extract_features.py

  Changes vs v2:
    + Hjorth parameters (mobility, complexity) — 16 features
    + Phase-Locking Value (PLV) per band pair  — 140 features
    + Sample entropy per channel               —   8 features
    Keeps coherence from v2                    — 140 features

  Features per window (368 total):
    [0:40]    Band powers      8 ch × 5 bands  (log scale)
    [40:56]   Band ratios      8 ch × 2 ratios
    [56:64]   Spectral entropy 8 ch
    [64:80]   Hjorth mobility  8 ch
    [80:96]   Hjorth complexity 8 ch
    [96:236]  Coherence        28 pairs × 5 bands
    [236:376] PLV              28 pairs × 5 bands
    [376:384] Permutation entropy 7 ch

  Reads : data/processed_v3/PNxx.npz
  Output: data/features/features_v3.npz
=============================================================
"""

import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, sosfilt
from itertools import combinations
from pathlib import Path

PROCESSED_DIR = Path("data/processed_v4")
FEATURES_DIR  = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SFREQ      = 250
N_CHANNELS = 7   # headband friendly: T3, T5, O1, Pz, O2, T6, T4
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
BAND_NAMES    = list(BANDS.keys())
N_BANDS       = len(BANDS)
CHANNEL_PAIRS = list(combinations(range(N_CHANNELS), 2))
N_PAIRS       = len(CHANNEL_PAIRS)   # 28


# Band powers + ratios + spectral entropy (64) 

def extract_band_features(window: np.ndarray) -> np.ndarray:
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
    sp_entr = -(pn * np.log(pn)).sum(axis=1)

    return np.concatenate([
        band_powers.flatten(),   # 40
        theta_alpha,             # 8
        delta_beta,              # 8
        sp_entr,                 # 8
    ]).astype(np.float32)        # total: 64


# Hjorth parameters (16) 

def extract_hjorth(window: np.ndarray) -> np.ndarray:
    """
    Hjorth mobility and complexity per channel.
    Mobility  = sqrt(var(x') / var(x))   — mean frequency proxy
    Complexity = mobility(x') / mobility(x) — bandwidth proxy
    """
    eps   = 1e-10
    d1    = np.diff(window, axis=-1)
    d2    = np.diff(d1,     axis=-1)

    var_x  = window[:, :-2].var(axis=-1)   # trim to match d2 length
    var_d1 = d1[:, :-1].var(axis=-1)
    var_d2 = d2.var(axis=-1)

    mobility   = np.sqrt(var_d1 / (var_x  + eps))
    complexity = np.sqrt(var_d2 / (var_d1 + eps)) / (mobility + eps)

    return np.concatenate([mobility, complexity]).astype(np.float32)  # 16


# Magnitude squared coherence (140)

def extract_coherence(window: np.ndarray) -> np.ndarray:
    coh_features = np.zeros((N_PAIRS, N_BANDS), dtype=np.float32)
    for p_idx, (i, j) in enumerate(CHANNEL_PAIRS):
        freqs, coh = signal.coherence(window[i], window[j],
                                      fs=SFREQ, nperseg=512)
        for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() > 0:
                coh_features[p_idx, b_idx] = coh[mask].mean()
    return coh_features.flatten()   # 140


# Phase locking value (140)

def extract_plv(window: np.ndarray) -> np.ndarray:
    """
    PLV measures phase synchrony between channel pairs independent of amplitude.
    PLV = |mean(exp(i * phase_diff))|, ranges 0 (no sync) to 1 (perfect sync).
    More sensitive than coherence for detecting preictal synchronization.
    """
    plv_features = np.zeros((N_PAIRS, N_BANDS), dtype=np.float32)
    for b_idx, (_, (lo, hi)) in enumerate(BANDS.items()):
        sos = butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
        # Filter and extract instantaneous phase for all channels at once
        phases = np.angle(hilbert(sosfilt(sos, window, axis=-1), axis=-1))
        for p_idx, (i, j) in enumerate(CHANNEL_PAIRS):
            phase_diff = phases[i] - phases[j]
            plv_features[p_idx, b_idx] = float(
                np.abs(np.mean(np.exp(1j * phase_diff)))
            )
    return plv_features.flatten()   # 140


# Permutation entropy (7) 

def permutation_entropy_channel(x: np.ndarray, order: int = 3,
                                 delay: int = 1) -> float:
    """
    O(N log N) complexity measure — same information as sample entropy
    (signal predictability decreases in preictal period) but ~100× faster.
    """
    N = len(x)
    counts = {}
    for i in range(N - delay * (order - 1)):
        pattern = tuple(np.argsort(x[i : i + delay * order : delay]))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def extract_permutation_entropy(window: np.ndarray) -> np.ndarray:
    return np.array([
        permutation_entropy_channel(window[ch])
        for ch in range(N_CHANNELS)
    ], dtype=np.float32)


# All features combined 

def extract_all_features(window: np.ndarray) -> np.ndarray:
    """
    Extract all 368 features for one window.

    [0:56]    band powers + ratios + spectral entropy
    [56:70]   Hjorth (mobility + complexity)
    [70:175]  coherence (21 pairs × 5 bands)
    [175:280] PLV      (21 pairs × 5 bands)
    [280:287] permutation entropy
    """
    return np.concatenate([
        extract_band_features(window),   # 64
        extract_hjorth(window),          # 16
        extract_coherence(window),       # 140
        extract_plv(window),             # 140
        extract_permutation_entropy(window),  # 7
    ]).astype(np.float32)


# Patient relative normalization 

def normalize_to_baseline(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    interictal_mask = y == 0
    n_inter = interictal_mask.sum()

    if n_inter < 10:
        print(f"    Only {n_inter} interictal windows — skipping normalization")
        return X

    baseline_mean = X[interictal_mask].mean(axis=0)
    baseline_std  = X[interictal_mask].std(axis=0)
    baseline_std[baseline_std < 1e-10] = 1.0

    return ((X - baseline_mean) / baseline_std).astype(np.float32)


# Main 

N_BP  = N_CHANNELS * N_BANDS + N_CHANNELS * 2 + N_CHANNELS   # 64
N_HJ  = N_CHANNELS * 2                                        # 16
N_COH = N_PAIRS * N_BANDS                                     # 140
N_PLV = N_PAIRS * N_BANDS                                     # 140
N_SE  = N_CHANNELS                                            # 7
N_TOT = N_BP + N_HJ + N_COH + N_PLV + N_SE                   # 368

print("\n" + "=" * 62)
print("  FEATURE EXTRACTION v3")
print("=" * 62)
print(f"  Band powers + ratios + spectral entropy : {N_BP}")
print(f"  Hjorth parameters                       : {N_HJ}")
print(f"  Coherence  ({N_PAIRS} pairs × {N_BANDS} bands)          : {N_COH}")
print(f"  PLV        ({N_PAIRS} pairs × {N_BANDS} bands)          : {N_PLV}")
print(f"  Sample entropy                          : {N_SE}")
print(f"  {'─'*40}")
print(f"  Total features                          : {N_TOT}")
print(f"\n  Source  : {PROCESSED_DIR}/")
print(f"  Output  : {FEATURES_DIR}/features_v3.npz")
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
    X_raw     = data["X"]   # (N, n_ch, 7500)
    y         = data["y"]
    n_windows = len(X_raw)
    n_pre     = int((y == 1).sum())
    n_inter   = int((y == 0).sum())

    print(f"── {patient}  ({n_windows} windows | pre: {n_pre} | inter: {n_inter})")

    X_feat = []
    for i in range(n_windows):
        X_feat.append(extract_all_features(X_raw[i]))
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{n_windows} windows...")

    X_feat = np.array(X_feat, dtype=np.float32)

    if not np.isfinite(X_feat).all():
        n_bad = (~np.isfinite(X_feat)).sum()
        print(f"   {n_bad} non-finite values — clipping")
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    X_norm = normalize_to_baseline(X_feat, y)

    print(f"   Normalized — mean: {X_norm.mean():.4f}  std: {X_norm.std():.4f}")

    all_X.append(X_norm)
    all_y.append(y)
    all_patients.extend([patient] * n_windows)

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
print(f"  Mean           : {X_all.mean():.4f}")
print(f"  Std            : {X_all.std():.4f}")

out_path = FEATURES_DIR / "features_v4.npz"
np.savez_compressed(str(out_path), X=X_all, y=y_all, patients=p_all)
print(f"\n  Saved → {out_path}")
print(f"{'=' * 62}\n")
