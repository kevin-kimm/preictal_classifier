"""
=============================================================
  Siena Scalp EEG — Model Training v3
  05_train_model.py

  Changes vs v2:
    + GradientBoosting with sample weights (no XGBoost needed)
    + Focal loss replaces binary_crossentropy for NN
      (down-weights easy interictal negatives)
    + Larger NN (256→128→64→32→1)

  Loads:   data/features/features_v3.npz
  Method:  Leave One Patient Out (LOPO) cross-validation
  Models:  1) GradientBoosting
           2) Dense neural network with focal loss

  Output:  models_v3/lopo_results.json
           models_v3/gb_<patient>.pkl
           models_v3/nn_<patient>.keras
=============================================================
"""

import json
import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
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

FEATURES_PATH = Path("data/features/features_v3.npz")
MODELS_DIR    = Path("models_v3")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 0.65
BATCH_SIZE = 64
EPOCHS     = 50


# Metrics
def evaluate(y_true, y_prob, threshold=THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "auc_roc":      float(roc_auc_score(y_true, y_prob)),
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }



# Gradient boosting
def build_gb() -> GradientBoostingClassifier:
    """
    sklearn GradientBoosting — no external dependencies needed.
    Imbalance handled via sample_weight during fit().
    """
    return GradientBoostingClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )


# Focal loss
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss down-weights easy examples (confident interictal
    windows) so training focuses on hard cases near the decision
    boundary.

    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class balance weight for the positive class
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        return alpha * tf.pow(1.0 - p_t, gamma) * bce
    return loss



# Dense neural network
def build_nn(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")

    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation="sigmoid",
                          name="preictal_prob")(x)

    return keras.Model(inputs=inputs, outputs=output)


# Load data
print("\n" + "=" * 62)
print("  LOADING FEATURES v3")
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
print(f"  Preictal   : {int((y_all==1).sum()):,}")
print(f"  Interictal : {int((y_all==0).sum()):,}")



# LOPO training
print("\n" + "=" * 62)
print("  LEAVE ONE PATIENT OUT — v3")
print("=" * 62)

gb_results = {}
nn_results = {}

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"\n── Fold {fold_idx+1}/{len(patient_ids)} "
          f"| Test: {test_patient} " + "─" * 25)

    test_mask  = patients == test_patient
    train_mask = ~test_mask

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    if (y_test == 1).sum() == 0:
        print("  No preictal windows in test set — skipping")
        continue

    n_pre   = int((y_train == 1).sum())
    n_inter = int((y_train == 0).sum())
    spw     = n_inter / n_pre

    print(f"  Train : {len(y_train):,} windows "
          f"(pre: {n_pre:,} | inter: {n_inter:,})  spw: {spw:.1f}")
    print(f"  Test  : {len(y_test):,} windows "
          f"(pre: {int((y_test==1).sum()):,} | "
          f"inter: {int((y_test==0).sum()):,})")

    # Scale features fit on train only
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # GradientBoosting 
    print(f"\n  [GradientBoosting] training...")
    gb_model      = build_gb()
    sample_weight = np.where(y_train == 1, spw, 1.0)
    gb_model.fit(X_train_scaled, y_train,
                 sample_weight=sample_weight)

    gb_prob    = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_metrics = evaluate(y_test, gb_prob)
    gb_results[test_patient] = gb_metrics

    print(f"  [GradientBoosting] AUC: {gb_metrics['auc_roc']:.3f}  "
          f"F1: {gb_metrics['f1']:.3f}  "
          f"Precision: {gb_metrics['precision']:.3f}  "
          f"Recall: {gb_metrics['recall']:.3f}")
    print(f"                     TP: {gb_metrics['tp']}  "
          f"FP: {gb_metrics['fp']}  "
          f"FN: {gb_metrics['fn']}  "
          f"TN: {gb_metrics['tn']}")

    joblib.dump(
        {"model": gb_model, "scaler": scaler},
        str(MODELS_DIR / f"gb_{test_patient}.pkl")
    )

    # Neural network with focal loss
    print(f"\n  [Neural net] training...")

    nn_model = build_nn(n_features)
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=7, restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=0,
        ),
    ]

    nn_model.fit(
        X_train_scaled, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.15,
        class_weight={0: 1.0, 1: spw},
        callbacks=callbacks,
        verbose=0,
    )

    nn_prob    = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_metrics = evaluate(y_test, nn_prob)
    nn_results[test_patient] = nn_metrics

    print(f"  [Neural net] AUC: {nn_metrics['auc_roc']:.3f}  "
          f"F1: {nn_metrics['f1']:.3f}  "
          f"Precision: {nn_metrics['precision']:.3f}  "
          f"Recall: {nn_metrics['recall']:.3f}")
    print(f"               TP: {nn_metrics['tp']}  "
          f"FP: {nn_metrics['fp']}  "
          f"FN: {nn_metrics['fn']}  "
          f"TN: {nn_metrics['tn']}")

    nn_model.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))



# Summary
def summarize(results: dict, model_name: str):
    if not results:
        return
    aucs  = [r["auc_roc"]   for r in results.values()]
    f1s   = [r["f1"]        for r in results.values()]
    precs = [r["precision"] for r in results.values()]
    recs  = [r["recall"]    for r in results.values()]

    print(f"\n{'=' * 62}")
    print(f"  {model_name} — LOPO SUMMARY")
    print(f"{'=' * 62}")
    print(f"  {'Patient':<10} {'AUC':>7} {'F1':>7} "
          f"{'Precision':>10} {'Recall':>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 58)
    for pat, r in sorted(results.items()):
        print(f"  {pat:<10} "
              f"{r['auc_roc']:>7.3f} "
              f"{r['f1']:>7.3f} "
              f"{r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} "
              f"{r['tp']:>5} "
              f"{r['fp']:>5} "
              f"{r['fn']:>5}")
    print("  " + "-" * 58)
    print(f"  {'MEAN':<10} "
          f"{np.mean(aucs):>7.3f} "
          f"{np.mean(f1s):>7.3f} "
          f"{np.mean(precs):>10.3f} "
          f"{np.mean(recs):>8.3f}")
    print(f"  {'STD':<10} "
          f"{np.std(aucs):>7.3f} "
          f"{np.std(f1s):>7.3f} "
          f"{np.std(precs):>10.3f} "
          f"{np.std(recs):>8.3f}")
    print(f"{'=' * 62}")


summarize(gb_results, "GRADIENT BOOSTING")
summarize(nn_results, "NEURAL NETWORK (focal loss)")

# ── Head-to-head ──────────────────────────────────────────
common = sorted(set(gb_results) & set(nn_results))
if common:
    print(f"\n{'=' * 62}")
    print(f"  Head-to-head: GradientBoosting vs Neural Network")
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

all_results = {"gradient_boosting": gb_results, "neural_net": nn_results}
with open(MODELS_DIR / "lopo_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved → {MODELS_DIR}/lopo_results.json\n")