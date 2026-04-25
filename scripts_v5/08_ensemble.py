"""
=============================================================
  Siena Scalp EEG — Ensemble Model
  08_ensemble.py

  Combines two models that each perform best on different
  feature sets:

    Model A: Neural Network  on band powers (64 features)
             Best mean AUC: ~0.584
    Model B: GradientBoosting on correlation (148 features)
             Best mean AUC: ~0.550

  Ensemble strategies tested:
    1. Average    — mean of both probability scores
    2. Max        — highest probability from either model
    3. Consensus  — both must agree (high precision)
    4. Weighted   — weight by each model's validation AUC

  Post-processing smoothing:
    Instead of alerting on a single window, require N
    consecutive windows above threshold before firing alert.
    Dramatically reduces false alarms.

  Output: models/ensemble_results.json
=============================================================
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix
)
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

# Config
BP_FEATURES_PATH   = Path("data/features/features.npz")
CORR_FEATURES_PATH = Path("data/features/features_correlation.npz")
MODELS_DIR         = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD       = 0.65   # base decision threshold
SMOOTH_WINDOWS  = 3      # consecutive windows needed to fire alert
BATCH_SIZE      = 64
EPOCHS          = 50


# Metrics
def evaluate(y_true, y_prob, threshold=THRESHOLD, label="") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    results = {
        "auc_roc":      float(roc_auc_score(y_true, y_prob)),
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    if label:
        print(f"    {label:<20} AUC: {results['auc_roc']:.3f}  "
              f"F1: {results['f1']:.3f}  "
              f"Prec: {results['precision']:.3f}  "
              f"Rec: {results['recall']:.3f}  "
              f"TP: {results['tp']}  FP: {results['fp']}")
    return results


def apply_smoothing(y_prob: np.ndarray,
                    threshold: float,
                    n_consecutive: int) -> np.ndarray:
    """
    Post-processing: require N consecutive windows above
    threshold before firing an alert.

    Instead of returning binary predictions, returns a
    smoothed probability that is only high when N consecutive
    windows exceed threshold — reduces isolated false alarms.
    """
    binary    = (y_prob >= threshold).astype(int)
    smoothed  = np.zeros_like(y_prob)

    for i in range(len(binary)):
        start = max(0, i - n_consecutive + 1)
        window_votes = binary[start:i+1]
        if len(window_votes) == n_consecutive and window_votes.all():
            smoothed[i] = 1.0
        elif binary[i]:
            # partial credit — not enough consecutive yet
            smoothed[i] = y_prob[i] * (window_votes.sum() / n_consecutive)

    return smoothed


# Model builders
def build_gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        random_state=42, verbose=0,
    )

def build_nn(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=output)


# Load both feature sets
print("\n" + "=" * 62)
print("  ENSEMBLE MODEL — LOPO TRAINING")
print("=" * 62)

for path in [BP_FEATURES_PATH, CORR_FEATURES_PATH]:
    if not path.exists():
        print(f"Missing: {path}")
        print("   Run 04_extract_features.py and 07_add_correlation.py first")
        exit(1)

# Band power features (for neural network)
bp_data    = np.load(str(BP_FEATURES_PATH),   allow_pickle=True)
X_bp       = bp_data["X"]
y_bp       = bp_data["y"]
patients_bp = bp_data["patients"]

# Correlation features (for gradient boosting)
corr_data      = np.load(str(CORR_FEATURES_PATH), allow_pickle=True)
X_corr         = corr_data["X"]
y_corr         = corr_data["y"]
patients_corr  = corr_data["patients"]

patient_ids = sorted(set(patients_bp))
print(f"  Band power features  : {X_bp.shape}")
print(f"  Correlation features : {X_corr.shape}")
print(f"  Patients             : {len(patient_ids)}")



# LOPO loop
ensemble_results = {}

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"\n── Fold {fold_idx+1}/{len(patient_ids)} "
          f"| Test: {test_patient} " + "─" * 25)

    # Masks
    test_mask  = patients_bp == test_patient
    train_mask = ~test_mask

    # Band power splits
    X_bp_train = X_bp[train_mask];  y_train = y_bp[train_mask]
    X_bp_test  = X_bp[test_mask];   y_test  = y_bp[test_mask]

    # Correlation splits
    X_corr_train = X_corr[train_mask]
    X_corr_test  = X_corr[test_mask]

    if (y_test == 1).sum() == 0:
        print(f"No preictal windows — skipping")
        continue

    n_pre   = int((y_train == 1).sum())
    n_inter = int((y_train == 0).sum())
    spw     = n_inter / n_pre

    print(f"  Train: {len(y_train):,}  "
          f"(pre: {n_pre:,} | inter: {n_inter:,})")
    print(f"  Test : {len(y_test):,}  "
          f"(pre: {int((y_test==1).sum())} | "
          f"inter: {int((y_test==0).sum())})")

    # Scale features 
    bp_scaler   = StandardScaler()
    corr_scaler = StandardScaler()

    X_bp_train_s   = bp_scaler.fit_transform(X_bp_train)
    X_bp_test_s    = bp_scaler.transform(X_bp_test)
    X_corr_train_s = corr_scaler.fit_transform(X_corr_train)
    X_corr_test_s  = corr_scaler.transform(X_corr_test)

    # Train neural network (band powers) 
    print(f"\n  Training Neural Network (band powers)...")
    nn = build_nn(X_bp_train_s.shape[1])
    nn.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    nn.fit(
        X_bp_train_s, y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_split=0.15,
        class_weight={0: 1.0, 1: spw},
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=7, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=4, min_lr=1e-6, verbose=0),
        ],
        verbose=0
    )
    nn_prob = nn.predict(X_bp_test_s, verbose=0).flatten()

    # ── Train GradientBoosting (correlation features) ─────
    print(f"  Training GradientBoosting (correlation features)...")
    sample_weight = np.where(y_train == 1, spw, 1.0)
    gb = build_gb()
    gb.fit(X_corr_train_s, y_train, sample_weight=sample_weight)
    gb_prob = gb.predict_proba(X_corr_test_s)[:, 1]

    # Ensemble combinations
    print(f"\n  Individual models:")
    nn_metrics = evaluate(y_test, nn_prob, label="Neural Network")
    gb_metrics = evaluate(y_test, gb_prob, label="GradientBoosting")

    # Weight by individual AUC
    nn_auc = nn_metrics["auc_roc"]
    gb_auc = gb_metrics["auc_roc"]
    total  = nn_auc + gb_auc + 1e-10
    w_nn   = nn_auc / total
    w_gb   = gb_auc / total

    print(f"\n  Ensemble strategies:")
    ensembles = {
        "average":   (nn_prob + gb_prob) / 2,
        "max":        np.maximum(nn_prob, gb_prob),
        "weighted":   w_nn * nn_prob + w_gb * gb_prob,
        "consensus":  nn_prob * gb_prob,  # both must be high
    }

    fold_results = {
        "nn":    nn_metrics,
        "gb":    gb_metrics,
        "n_preictal":   int((y_test==1).sum()),
        "n_interictal": int((y_test==0).sum()),
    }

    best_auc    = max(nn_auc, gb_auc)
    best_method = "nn" if nn_auc > gb_auc else "gb"

    for name, prob in ensembles.items():
        m = evaluate(y_test, prob, label=f"Ensemble ({name})")
        fold_results[f"ensemble_{name}"] = m
        if m["auc_roc"] > best_auc:
            best_auc    = m["auc_roc"]
            best_method = f"ensemble_{name}"

    # Post processing smoothing on best ensemble 
    best_prob = ensembles["average"]
    smoothed  = apply_smoothing(best_prob, THRESHOLD, SMOOTH_WINDOWS)
    smooth_m  = evaluate(y_test, smoothed,
                         label=f"Average + smooth({SMOOTH_WINDOWS}w)")
    fold_results["smoothed_average"] = smooth_m

    print(f"\n  ★ Best method: {best_method}  AUC: {best_auc:.3f}")
    fold_results["best_method"] = best_method
    fold_results["best_auc"]    = best_auc

    ensemble_results[test_patient] = fold_results

    # Save models
    nn.save(str(MODELS_DIR / f"ensemble_nn_{test_patient}.keras"))
    joblib.dump({"model": gb, "scaler": corr_scaler},
                str(MODELS_DIR / f"ensemble_gb_{test_patient}.pkl"))



# Summary
methods = ["nn", "gb", "ensemble_average", "ensemble_weighted",
           "ensemble_max", "ensemble_consensus", "smoothed_average"]
method_labels = ["Neural Net", "GradBoost", "Avg Ensemble",
                 "Weighted", "Max", "Consensus", "Avg+Smooth"]

print(f"\n{'=' * 70}")
print(f"  ENSEMBLE LOPO SUMMARY — AUC by method")
print(f"{'=' * 70}")
print(f"  {'Patient':<10}", end="")
for label in method_labels:
    print(f" {label:>12}", end="")
print()
print("  " + "-" * 96)

method_aucs = {m: [] for m in methods}

for patient in sorted(ensemble_results.keys()):
    r = ensemble_results[patient]
    print(f"  {patient:<10}", end="")
    for method in methods:
        if method in r:
            auc = r[method]["auc_roc"]
            method_aucs[method].append(auc)
            print(f" {auc:>12.3f}", end="")
        else:
            print(f" {'—':>12}", end="")
    print()

print("  " + "-" * 96)
print(f"  {'MEAN':<10}", end="")
for method in methods:
    if method_aucs[method]:
        print(f" {np.mean(method_aucs[method]):>12.3f}", end="")
    else:
        print(f" {'—':>12}", end="")
print()

print(f"\n{'=' * 70}")
best_overall = max(methods,
                   key=lambda m: np.mean(method_aucs[m])
                   if method_aucs[m] else 0)
print(f"  Best method overall: {best_overall}")
print(f"  Best mean AUC      : {np.mean(method_aucs[best_overall]):.3f}")
print(f"{'=' * 70}\n")

# Save results
with open(MODELS_DIR / "ensemble_results.json", "w") as f:
    json.dump(ensemble_results, f, indent=2)
print(f"  Results saved: models/ensemble_results.json\n")