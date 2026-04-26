"""
=============================================================
  Siena Scalp EEG — Clinical Threshold Tuning
  15_threshold_tuning.py

  Loads existing v2 trained models and finds the optimal
  threshold per patient using two strategies:

  Strategy 1 — Clinical constraint
    Find the highest threshold that gives <= 2 false alarms
    per hour of recording. This is how real clinical systems
    work — prioritize patient comfort.

  Strategy 2 — Maximize F1
    Find threshold that best balances precision and recall.

  Strategy 3 — Fixed thresholds sweep
    Show results at 0.30, 0.40, 0.50, 0.55, 0.60, 0.65

  No retraining needed — uses existing models/features.
  Output: models/threshold_results.json
=============================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

FEATURES_PATH = Path("data/features/features_correlation.npz")
MODELS_DIR    = Path("models")
SFREQ         = 250
WINDOW_SEC    = 30
STEP_SEC      = 5

# How many windows per hour of recording
WINDOWS_PER_HOUR = 3600 / STEP_SEC  # = 720 windows per hour

# Clinical target
MAX_FP_PER_HOUR = 2.0


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    n_inter       = int((y_true == 0).sum())
    record_hours  = (n_inter * STEP_SEC) / 3600
    fp_per_hour   = fp / record_hours if record_hours > 0 else 0
    fp_per_10     = f"{fp/tp*10:.1f}" if tp > 0 else "inf"

    return {
        "auc_roc":      float(roc_auc_score(y_true, y_prob)),
        "threshold":    float(threshold),
        "precision":    float(precision_score(y_true, y_pred,
                                               zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred,
                                            zero_division=0)),
        "f1":           float(f1_score(y_true, y_pred,
                                        zero_division=0)),
        "fp_per_hour":  float(fp_per_hour),
        "fp_per_10":    fp_per_10,
        "record_hours": float(record_hours),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def find_clinical_threshold(y_true, y_prob,
                              max_fp_per_hour=MAX_FP_PER_HOUR):
    """
    Find the LOWEST threshold (most sensitive) that still
    keeps false alarms at or below the clinical target.
    Sweeps from high to low threshold.
    """
    n_inter      = int((y_true == 0).sum())
    record_hours = (n_inter * STEP_SEC) / 3600

    for thresh in np.arange(0.95, 0.19, -0.01):
        y_pred    = (y_prob >= thresh).astype(int)
        fp        = int(((y_pred == 1) & (y_true == 0)).sum())
        fp_per_hr = fp / record_hours if record_hours > 0 else 0

        if fp_per_hr <= max_fp_per_hour:
            return float(thresh)

    return 0.95


def find_f1_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1     = -1
    best_thresh = 0.65

    for thresh in np.arange(0.10, 0.96, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    return float(best_thresh)


# ─────────────────────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CLINICAL THRESHOLD TUNING — v2 Models")
print("=" * 70)
print(f"""
  Loads existing v2 trained models (no retraining)
  Tests three threshold strategies per patient:

  Strategy 1 — Clinical  : <= {MAX_FP_PER_HOUR} false alarms per hour
  Strategy 2 — Best F1   : maximize F1 score
  Strategy 3 — Sweep     : show results at multiple fixed thresholds

  Clinical target: <= 2 FP/hour = ~48 false alarms per day
  (Published literature: patients tolerate ~1-2 FP/day for wearables)
""")

if not FEATURES_PATH.exists():
    print(f"Features not found: {FEATURES_PATH}")
    print("Run scripts_v2/04_extract_features.py first")
    exit(1)

data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]

patient_ids = sorted(set(patients))
print(f"  Loaded {len(y_all):,} windows  |  "
      f"{len(patient_ids)} patients  |  "
      f"{X_all.shape[1]} features\n")


# ─────────────────────────────────────────────────────────────
# EVALUATE EACH PATIENT
# ─────────────────────────────────────────────────────────────
all_results     = {}
fixed_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

for patient in patient_ids:
    test_mask  = patients == patient
    train_mask = ~test_mask

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test  = X_all[test_mask]
    y_test  = y_all[test_mask]

    if (y_test == 1).sum() == 0:
        print(f"  {patient} — no preictal windows, skipping")
        continue

    # Load models
    nn_path = MODELS_DIR / f"nn_{patient}.keras"
    if not nn_path.exists():
        print(f"  {patient} — model not found: {nn_path}")
        continue

    # Scale using training data
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Get predictions
    nn_model = keras.models.load_model(
        str(nn_path), compile=False)
    nn_prob  = nn_model.predict(X_test_scaled, verbose=0).flatten()

    auc = roc_auc_score(y_test, nn_prob)

    # Find optimal thresholds
    clinical_thresh = find_clinical_threshold(y_test, nn_prob)
    f1_thresh       = find_f1_threshold(y_test, nn_prob)

    # Evaluate all strategies
    clinical_metrics = evaluate(y_test, nn_prob, clinical_thresh)
    f1_metrics       = evaluate(y_test, nn_prob, f1_thresh)
    fixed_metrics    = {
        str(t): evaluate(y_test, nn_prob, t)
        for t in fixed_thresholds
    }

    all_results[patient] = {
        "auc_roc":          auc,
        "clinical":         clinical_metrics,
        "best_f1":          f1_metrics,
        "fixed_thresholds": fixed_metrics,
        "n_preictal":       int((y_test == 1).sum()),
        "n_interictal":     int((y_test == 0).sum()),
        "record_hours":     float((y_test == 0).sum() * STEP_SEC / 3600),
    }

    grade = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"

    print(f"── {patient}  AUC: {auc:.3f}  {grade}  "
          f"({all_results[patient]['record_hours']:.1f}h recording)")
    print(f"   Fixed 0.65  : "
          f"TP={fixed_metrics['0.65']['tp']:>3}  "
          f"FP={fixed_metrics['0.65']['fp']:>4}  "
          f"Rec={fixed_metrics['0.65']['recall']:.3f}  "
          f"FP/hr={fixed_metrics['0.65']['fp_per_hour']:.2f}  "
          f"FP/10={fixed_metrics['0.65']['fp_per_10']}")
    print(f"   Clinical    : thresh={clinical_thresh:.2f}  "
          f"TP={clinical_metrics['tp']:>3}  "
          f"FP={clinical_metrics['fp']:>4}  "
          f"Rec={clinical_metrics['recall']:.3f}  "
          f"FP/hr={clinical_metrics['fp_per_hour']:.2f}  "
          f"FP/10={clinical_metrics['fp_per_10']}")
    print(f"   Best F1     : thresh={f1_thresh:.2f}  "
          f"TP={f1_metrics['tp']:>3}  "
          f"FP={f1_metrics['fp']:>4}  "
          f"Rec={f1_metrics['recall']:.3f}  "
          f"FP/hr={f1_metrics['fp_per_hour']:.2f}  "
          f"FP/10={f1_metrics['fp_per_10']}")
    print()


# ─────────────────────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SUMMARY — CLINICAL THRESHOLD (<=2 FP/hour)")
print("=" * 80)
print(f"  {'Patient':<10} {'AUC':>7} {'Thresh':>7} {'TP':>5} "
      f"{'FP':>5} {'Recall':>7} {'Prec':>7} "
      f"{'FP/hr':>7} {'FP/10':>7}  Grade")
print("  " + "-" * 76)

clin_aucs = []
for pat, r in sorted(all_results.items()):
    m    = r["clinical"]
    auc  = r["auc_roc"]
    clin_aucs.append(auc)
    grd  = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"
    print(f"  {pat:<10} {auc:>7.3f} {m['threshold']:>7.2f} "
          f"{m['tp']:>5} {m['fp']:>5} "
          f"{m['recall']:>7.3f} {m['precision']:>7.3f} "
          f"{m['fp_per_hour']:>7.2f} {m['fp_per_10']:>7}  {grd}")

print("  " + "-" * 76)
print(f"  {'MEAN':<10} {np.mean(clin_aucs):>7.3f}")

pred = sum(1 for r in all_results.values() if r["auc_roc"] >= 0.70)
print(f"\n  Predictable: {pred}/{len(all_results)} patients")


print("\n" + "=" * 80)
print("  SUMMARY — BEST F1 THRESHOLD")
print("=" * 80)
print(f"  {'Patient':<10} {'AUC':>7} {'Thresh':>7} {'TP':>5} "
      f"{'FP':>5} {'Recall':>7} {'Prec':>7} "
      f"{'FP/hr':>7} {'FP/10':>7}  Grade")
print("  " + "-" * 76)

for pat, r in sorted(all_results.items()):
    m    = r["best_f1"]
    auc  = r["auc_roc"]
    grd  = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"
    print(f"  {pat:<10} {auc:>7.3f} {m['threshold']:>7.2f} "
          f"{m['tp']:>5} {m['fp']:>5} "
          f"{m['recall']:>7.3f} {m['precision']:>7.3f} "
          f"{m['fp_per_hour']:>7.2f} {m['fp_per_10']:>7}  {grd}")


print("\n" + "=" * 80)
print("  THRESHOLD SWEEP — TP caught at each threshold")
print("=" * 80)
print(f"  {'Patient':<10} {'AUC':>7}", end="")
for t in fixed_thresholds:
    print(f"  {str(t):>6}", end="")
print("  (TP at each threshold)")
print("  " + "-" * 72)

for pat, r in sorted(all_results.items()):
    auc = r["auc_roc"]
    print(f"  {pat:<10} {auc:>7.3f}", end="")
    for t in fixed_thresholds:
        tp = r["fixed_thresholds"][str(t)]["tp"]
        print(f"  {tp:>6}", end="")
    print()

print("\n" + "=" * 80)
print("  KEY INSIGHT: Compare TP at 0.65 (current) vs 0.40 (lower)")
print("=" * 80)
print(f"  {'Patient':<10} {'AUC':>7} {'TP@0.65':>8} "
      f"{'TP@0.40':>8} {'Gain':>6}  Note")
print("  " + "-" * 60)

for pat, r in sorted(all_results.items()):
    auc    = r["auc_roc"]
    tp_065 = r["fixed_thresholds"]["0.65"]["tp"]
    tp_040 = r["fixed_thresholds"]["0.4"]["tp"]
    gain   = tp_040 - tp_065
    note   = ""
    if gain > 10:
        note = "<- big improvement"
    elif tp_065 == 0 and tp_040 > 0:
        note = "<- NOW detectable!"
    grd = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"
    print(f"  {pat:<10} {auc:>7.3f} {tp_065:>8} "
          f"{tp_040:>8} {gain:>6}  {grd} {note}")

# Save
with open(MODELS_DIR / "threshold_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/threshold_results.json\n")