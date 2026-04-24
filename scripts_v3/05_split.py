"""
=============================================================
  Siena Scalp EEG — LOPO Split  v3
  05_split.py

  Loads features_v3.npz and produces one .npz per LOPO fold
  containing already-scaled train / val / test arrays.

  Reads : data/features/features_v3.npz
  Output: data/split/fold_{patient}.npz  — one per fold
          data/split/manifest.json       — fold inventory
=============================================================
"""

import json
import joblib
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path("data/features/features_v3.npz")
SPLIT_DIR     = Path("data/split")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO    = 0.15
RANDOM_STATE = 42


# Load features
print("\n" + "=" * 62)
print("  LOPO SPLIT  v3")
print("=" * 62)

if not FEATURES_PATH.exists():
    print(f"Features not found at {FEATURES_PATH}")
    print("Run 04_extract_features.py first.")
    exit(1)

data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]

patient_ids = sorted(set(patients))
n_features  = X_all.shape[1]

print(f"  Windows    : {len(y_all):,}")
print(f"  Features   : {n_features}")
print(f"  Patients   : {len(patient_ids)}")
print(f"  Preictal   : {int((y_all == 1).sum()):,}")
print(f"  Interictal : {int((y_all == 0).sum()):,}")
print(f"  Val ratio  : {VAL_RATIO}")
print()


# LOPO loop
saved_folds = []

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"── Fold {fold_idx+1}/{len(patient_ids)} | Test: {test_patient}")

    train_mask = patients != test_patient
    test_mask  = patients == test_patient

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    n_pre_test = int((y_test == 1).sum())
    if n_pre_test == 0:
        print(f"  No preictal windows in test — skipping\n")
        continue

    # Fit scaler on all training patients
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Stratified train / val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=VAL_RATIO,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    n_pre_tr  = int((y_tr  == 1).sum())
    n_pre_val = int((y_val == 1).sum())

    print(f"  train : {len(y_tr):>5,} windows  (pre: {n_pre_tr:>4,} | "
          f"inter: {int((y_tr==0).sum()):>5,}  ratio: {int((y_tr==0).sum())/n_pre_tr:.1f}:1)")
    print(f"  val   : {len(y_val):>5,} windows  (pre: {n_pre_val:>4,} | "
          f"inter: {int((y_val==0).sum()):>5,})")
    print(f"  test  : {len(y_test):>5,} windows  (pre: {n_pre_test:>4,} | "
          f"inter: {int((y_test==0).sum()):>5,})")

    # Save fold arrays
    fold_path = SPLIT_DIR / f"fold_{test_patient}.npz"
    np.savez_compressed(
        str(fold_path),
        X_tr=X_tr,   y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        X_test=X_test_scaled, y_test=y_test,
    )

    # Save scaler so it can be reused on live data
    joblib.dump(scaler, SPLIT_DIR / f"scaler_{test_patient}.pkl")

    saved_folds.append(test_patient)
    print(f"  Saved → {fold_path}\n")


# Manifest
manifest = {
    "features_path": str(FEATURES_PATH),
    "n_features":    n_features,
    "val_ratio":     VAL_RATIO,
    "random_state":  RANDOM_STATE,
    "folds":         saved_folds,
}
manifest_path = SPLIT_DIR / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("=" * 62)
print(f"  {len(saved_folds)} folds saved → {SPLIT_DIR}/")
print(f"  Manifest   → {manifest_path}")
print("=" * 62 + "\n")
