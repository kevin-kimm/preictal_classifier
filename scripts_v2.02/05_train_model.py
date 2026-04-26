"""
=============================================================
  Siena Scalp EEG — Model Training v2.02 FINAL
  05_train_model.py

  Optimal single-run configuration:
    Channels     : F7, T3, T5, C3, F8, T4, T6, C4
    Features     : 64 band powers
    Preictal     : 8 minutes (60% more data than v2 original)
    Step         : 5 seconds (independent windows)
    Loss         : Binary crossentropy + class weights
    Architecture : v2 proven 128->64->32->1
    Threshold    : 0.50
    Patience     : 12 epochs
    Epochs       : 100 max

  Why single run (no ensemble):
    Ensemble averaging low-AUC runs hurts performance
    With more training data, single run is more stable
=============================================================
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

FEATURES_PATH = Path("data/features/features_v202.npz")
MODELS_DIR    = Path("models_v202")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 0.50
BATCH_SIZE = 256
EPOCHS     = 100
STEP_SEC   = 5


def evaluate(y_true, y_prob, threshold=THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    n_inter      = int((y_true == 0).sum())
    record_hours = (n_inter * STEP_SEC) / 3600
    fp_per_hour  = fp / record_hours if record_hours > 0 else 0
    fp_per_10    = f"{fp/tp*10:.1f}" if tp > 0 else "inf"
    return {
        "auc_roc":     float(roc_auc_score(y_true, y_prob)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred,
                                              zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred,
                                           zero_division=0)),
        "fp_per_hour": float(fp_per_hour),
        "fp_per_10":   fp_per_10,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }


def build_nn(n_features):
    """
    v2 proven architecture — 128->64->32->1
    Binary crossentropy for stable training.
    """
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=output)


print("\n" + "=" * 66)
print("  v2.02 FINAL — OPTIMAL SINGLE RUN")
print("=" * 66)
print(f"""
  Channels     : F7, T3, T5, C3, F8, T4, T6, C4
  Features     : 64 band powers
  Preprocessing: notch + CAR + artifact rejection
  Preictal     : 8 min (60% more data than v2 original)
  Step         : 5s (independent windows, no overlap issue)
  Loss         : Binary crossentropy + class weights
  Architecture : 128->64->32->1
  Threshold    : {THRESHOLD}
  Batch size   : {BATCH_SIZE} (M4 optimized)
  Max epochs   : {EPOCHS}  patience=12
""")

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
print(f"  Features   : {n_features}")
print(f"  Patients   : {len(patient_ids)}")
print(f"  Preictal   : {int((y_all==1).sum()):,}")
print(f"  Interictal : {int((y_all==0).sum()):,}")

print("\n" + "=" * 66)
print("  LEAVE ONE PATIENT OUT")
print("=" * 66)

nn_results = {}

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"\n-- Fold {fold_idx+1}/{len(patient_ids)} "
          f"| Test: {test_patient} " + "-" * 28)

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
          f"(pre:{n_pre:,} | inter:{n_inter:,})  spw:{spw:.1f}")
    print(f"  Test  : {len(y_test):,} "
          f"(pre:{int((y_test==1).sum()):,} | "
          f"inter:{int((y_test==0).sum()):,})")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"  [Neural net] training (Metal GPU)...")

    nn = build_nn(n_features)
    nn.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    nn.fit(
        X_train_scaled, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.15,
        class_weight={0: 1.0, 1: spw},
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=12,
                restore_best_weights=True,
                verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0),
        ],
        verbose=0,
    )

    nn_prob    = nn.predict(X_test_scaled, verbose=0).flatten()
    metrics    = evaluate(y_test, nn_prob)
    nn_results[test_patient] = metrics

    grd = "✅" if metrics["auc_roc"] >= 0.70 else \
          "⚠️ " if metrics["auc_roc"] >= 0.60 else "❌"
    print(f"  {grd} AUC:{metrics['auc_roc']:.3f}  "
          f"Prec:{metrics['precision']:.3f}  "
          f"Rec:{metrics['recall']:.3f}  "
          f"TP:{metrics['tp']}  FP:{metrics['fp']}  "
          f"FP/hr:{metrics['fp_per_hour']:.1f}  "
          f"FP/10:{metrics['fp_per_10']}")

    nn.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))
    joblib.dump(scaler,
                str(MODELS_DIR / f"scaler_{test_patient}.pkl"))


# ── Summary ───────────────────────────────────────────────
print(f"\n{'=' * 72}")
print(f"  FINAL RESULTS  (threshold={THRESHOLD})")
print(f"{'=' * 72}")
print(f"  {'Patient':<10} {'AUC':>7} {'Prec':>7} {'Rec':>7} "
      f"{'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
      f"{'FP/hr':>7} {'FP/10':>7}  Grade")
print("  " + "-" * 76)

aucs = []
for pat, r in sorted(nn_results.items()):
    auc = r["auc_roc"]
    aucs.append(auc)
    grd = "✅ YES" if auc >= 0.70 else \
          "⚠️  MOD" if auc >= 0.60 else "❌ NO"
    print(f"  {pat:<10} {auc:>7.3f} {r['precision']:>7.3f} "
          f"{r['recall']:>7.3f} {r['f1']:>7.3f} "
          f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
          f"{r['fp_per_hour']:>7.1f} {r['fp_per_10']:>7}  {grd}")

print("  " + "-" * 76)
print(f"  {'MEAN':<10} {np.mean(aucs):>7.3f}")
print(f"{'=' * 72}")

pred  = sum(1 for a in aucs if a >= 0.70)
mod   = sum(1 for a in aucs if 0.60 <= a < 0.70)
poor  = sum(1 for a in aucs if a < 0.60)
total = len(aucs)

print(f"\n  ✅ Predictable (AUC >= 0.70) : {pred}/{total}")
print(f"  ⚠️  Modest     (0.60-0.70)   : {mod}/{total}")
print(f"  ❌ Poor        (< 0.60)      : {poor}/{total}")
print(f"\n  v2 original best             : 0.584 mean, ~6 predictable")
print(f"  v2.02 target                 : 7+ predictable")

with open(MODELS_DIR / "lopo_results.json", "w") as f:
    json.dump({"neural_net": nn_results,
               "config": {
                   "threshold":   THRESHOLD,
                   "preictal_min": 8,
                   "step_sec":    5,
                   "channels":    ["F7","T3","T5","C3",
                                    "F8","T4","T6","C4"],
                   "features":    64,
                   "loss":        "binary_crossentropy",
               }}, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/lopo_results.json\n")