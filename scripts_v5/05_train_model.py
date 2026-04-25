"""
=============================================================
  Siena Scalp EEG — Model Training v5
  05_train_model.py

  Best generalized detector:
    + v2 best ML channels (F7,T3,T5,C3,F8,T4,T6,C4)
    + v3 preprocessing quality (notch, CAR, artifact rejection)
    + 69 features (64 band powers + 5 asymmetry)
    + Focal loss alpha=0.75 (from v3 improvement)
    + Per-patient threshold tuning (NEW)
    + Metal GPU acceleration

  Per-patient threshold tuning:
    Instead of fixed 0.65 for everyone, finds the threshold
    per patient that maximizes F1 on the training data.
    This is the key improvement for generalization.

  Output: models_v5/lopo_results.json
=============================================================
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

FEATURES_PATH = Path("data/features/features_v5.npz")
MODELS_DIR    = Path("models_v5")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
EPOCHS     = 50

# Threshold search range
THRESHOLD_CANDIDATES = np.arange(0.30, 0.80, 0.05)


# PER-PATIENT THRESHOLD TUNING

def find_best_threshold(y_true: np.ndarray,
                         y_prob: np.ndarray) -> float:
    """
    Find the threshold that maximizes F1 score on validation data.
    This replaces the fixed 0.65 threshold with a patient-adaptive one.
    """
    best_f1   = -1
    best_thresh = 0.50

    for thresh in THRESHOLD_CANDIDATES:
        y_pred = (y_prob >= thresh).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    return float(best_thresh)



# METRICS
def evaluate(y_true, y_prob, threshold=0.65) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    return {
        "auc_roc":      float(roc_auc_score(y_true, y_prob)),
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":    float(precision_score(y_true, y_pred,
                                               zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred,
                                            zero_division=0)),
        "threshold":    float(threshold),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }



# GRADIENT BOOSTING
def build_gb():
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )


# FOCAL LOSS
def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1-y_true) * (1-y_pred)
        alpha_t = y_true * alpha + (1-y_true) * (1-alpha)
        return alpha_t * tf.pow(1-p_t, gamma) * bce
    return loss



# NEURAL NETWORK
def build_nn(n_features):
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=output)



# LOAD DATA
print("\n" + "=" * 62)
print("  LOADING FEATURES v5")
print("=" * 62)

if not FEATURES_PATH.exists():
    print(f"Features not found: {FEATURES_PATH}")
    print("Run 04_extract_features.py first.")
    exit(1)

data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]

patient_ids = sorted(set(patients))
n_features  = X_all.shape[1]

print(f"  Windows    : {len(y_all):,}")
print(f"  Features   : {n_features}  (64 band powers + 5 asymmetry)")
print(f"  Patients   : {len(patient_ids)}")
print(f"  Preictal   : {int((y_all==1).sum()):,}")
print(f"  Interictal : {int((y_all==0).sum()):,}")
print(f"  Batch size : {BATCH_SIZE} (M4 optimized)")
print(f"  Thresholds : per-patient tuned {THRESHOLD_CANDIDATES[0]:.2f}"
      f"–{THRESHOLD_CANDIDATES[-1]:.2f}")


# LOPO TRAINING LOOP
print("\n" + "=" * 62)
print("  LEAVE ONE PATIENT OUT — v5")
print("=" * 62)

gb_results = {}
nn_results = {}

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"\n-- Fold {fold_idx+1}/{len(patient_ids)} "
          f"| Test: {test_patient} " + "-" * 25)

    test_mask  = patients == test_patient
    train_mask = ~test_mask

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    if (y_test == 1).sum() == 0:
        print("  No preictal windows -- skipping")
        continue

    n_pre   = int((y_train == 1).sum())
    n_inter = int((y_train == 0).sum())
    spw     = n_inter / n_pre

    print(f"  Train : {len(y_train):,} "
          f"(pre: {n_pre:,} | inter: {n_inter:,})")
    print(f"  Test  : {len(y_test):,} "
          f"(pre: {int((y_test==1).sum()):,} | "
          f"inter: {int((y_test==0).sum()):,})")

    # Scale
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Use 15% of training for threshold tuning
    val_size    = max(int(0.15 * len(y_train)), 100)
    X_val_tune  = X_train_scaled[-val_size:]
    y_val_tune  = y_train[-val_size:]
    X_tr        = X_train_scaled[:-val_size]
    y_tr        = y_train[:-val_size]

    # GradientBoosting 
    print(f"\n  [GradientBoosting] training...")
    gb_model      = build_gb()
    sample_weight = compute_sample_weight("balanced", y_tr)
    gb_model.fit(X_tr, y_tr, sample_weight=sample_weight)

    gb_val_prob = gb_model.predict_proba(X_val_tune)[:, 1]
    gb_thresh   = find_best_threshold(y_val_tune, gb_val_prob)
    gb_prob     = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_metrics  = evaluate(y_test, gb_prob, threshold=gb_thresh)
    gb_results[test_patient] = gb_metrics

    print(f"  [GradientBoosting] threshold: {gb_thresh:.2f}  "
          f"AUC: {gb_metrics['auc_roc']:.3f}  "
          f"F1: {gb_metrics['f1']:.3f}  "
          f"Prec: {gb_metrics['precision']:.3f}  "
          f"Rec: {gb_metrics['recall']:.3f}")
    print(f"                     TP: {gb_metrics['tp']}  "
          f"FP: {gb_metrics['fp']}  "
          f"FN: {gb_metrics['fn']}")

    joblib.dump({"model": gb_model, "scaler": scaler,
                 "threshold": gb_thresh},
                str(MODELS_DIR / f"gb_{test_patient}.pkl"))

    # Neural network 
    print(f"\n  [Neural net] training (Metal GPU)...")

    nn_model = build_nn(n_features)
    nn_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[keras.metrics.AUC(name="auc")],
    )

    nn_model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_tune, y_val_tune),
        class_weight={0: 1.0, 1: spw},
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=7, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=4, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )

    nn_val_prob = nn_model.predict(X_val_tune, verbose=0).flatten()
    nn_thresh   = find_best_threshold(y_val_tune, nn_val_prob)
    nn_prob     = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_metrics  = evaluate(y_test, nn_prob, threshold=nn_thresh)
    nn_results[test_patient] = nn_metrics

    print(f"  [Neural net]       threshold: {nn_thresh:.2f}  "
          f"AUC: {nn_metrics['auc_roc']:.3f}  "
          f"F1: {nn_metrics['f1']:.3f}  "
          f"Prec: {nn_metrics['precision']:.3f}  "
          f"Rec: {nn_metrics['recall']:.3f}")
    print(f"                     TP: {nn_metrics['tp']}  "
          f"FP: {nn_metrics['fp']}  "
          f"FN: {nn_metrics['fn']}")

    nn_model.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))



# SUMMARY
def fp_per_10(tp, fp):
    if tp == 0: return "inf"
    return f"{fp/tp*10:.1f}"

def summarize(results, model_name):
    if not results: return
    aucs   = [r["auc_roc"]   for r in results.values()]
    f1s    = [r["f1"]        for r in results.values()]
    precs  = [r["precision"] for r in results.values()]
    recs   = [r["recall"]    for r in results.values()]
    thresh = [r["threshold"] for r in results.values()]

    print(f"\n{'=' * 70}")
    print(f"  {model_name} — LOPO SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Patient':<10} {'AUC':>7} {'Thresh':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>5} "
          f"{'FN':>5} {'FP/10':>7}  Grade")
    print("  " + "-" * 72)

    for pat, r in sorted(results.items()):
        auc  = r["auc_roc"]
        grd  = "YES" if auc >= 0.70 else "MOD" if auc >= 0.60 else "NO"
        fp10 = fp_per_10(r["tp"], r["fp"])
        print(f"  {pat:<10} {auc:>7.3f} {r['threshold']:>7.2f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['tp']:>5} {r['fp']:>5} "
              f"{r['fn']:>5} {fp10:>7}  {grd}")

    print("  " + "-" * 72)
    print(f"  {'MEAN':<10} {np.mean(aucs):>7.3f} "
          f"{np.mean(thresh):>7.2f} "
          f"{np.mean(precs):>7.3f} {np.mean(recs):>7.3f} "
          f"{np.mean(f1s):>7.3f}")
    print(f"{'=' * 70}")

    pred  = sum(1 for r in results.values() if r["auc_roc"] >= 0.70)
    mod   = sum(1 for r in results.values()
                if 0.60 <= r["auc_roc"] < 0.70)
    poor  = sum(1 for r in results.values() if r["auc_roc"] < 0.60)
    total = len(results)
    print(f"\n  Predictable (>= 0.70) : {pred}/{total}")
    print(f"  Modest (0.60-0.70)    : {mod}/{total}")
    print(f"  Poor   (< 0.60)       : {poor}/{total}")


summarize(gb_results, "GRADIENT BOOSTING")
summarize(nn_results, "NEURAL NETWORK (focal loss + per-patient threshold)")

# Head to head
common = sorted(set(gb_results) & set(nn_results))
if common:
    print(f"\n{'=' * 62}")
    print(f"  Head-to-head: GradBoost vs Neural Network")
    print(f"{'=' * 62}")
    print(f"  {'Patient':<10} {'GB AUC':>9} {'NN AUC':>9} {'Winner':>12}")
    print("  " + "-" * 44)
    gb_wins = nn_wins = 0
    for pat in common:
        ga = gb_results[pat]["auc_roc"]
        na = nn_results[pat]["auc_roc"]
        winner = "GradBoost" if ga > na else "Neural net"
        if ga > na: gb_wins += 1
        else: nn_wins += 1
        print(f"  {pat:<10} {ga:>9.3f} {na:>9.3f} {winner:>12}")
    print("  " + "-" * 44)
    print(f"  GradBoost wins: {gb_wins}  |  Neural net wins: {nn_wins}")
    print(f"{'=' * 62}")

all_results = {"gradient_boosting": gb_results,
               "neural_net": nn_results}
with open(MODELS_DIR / "lopo_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved -> {MODELS_DIR}/lopo_results.json\n")