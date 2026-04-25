"""
=============================================================
  Siena Scalp EEG — Model Training  v3
  06_train_model.py

  Loads pre-split folds from data/split/ and trains one
  GradientBoosting and one Neural Network per fold.
  Evaluates on train and val splits and saves metrics.

  Reads : data/split/fold_{patient}.npz
          data/split/manifest.json
  Output: models_v3/gb_{patient}.pkl
          models_v3/nn_{patient}.keras
          models_v3/train_metrics.json
=============================================================
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

# Enable Metal GPU acceleration on Apple Silicon M-series chips
# No effect on non-Apple hardware — safe to leave in
os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

SPLIT_DIR  = Path("data/split")
MODELS_DIR = Path("models_v3")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR = MODELS_DIR / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 0.65
BATCH_SIZE = 256    # optimized for M4 unified memory
EPOCHS     = 50


def evaluate(y_true, y_prob, threshold=THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "auc_pr":    float(average_precision_score(y_true, y_prob)),
        "loss":      float(log_loss(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }


def save_cm(m: dict, patient: str, model: str, split: str):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Interictal", "Preictal"])
    ax.set_yticklabels(["Interictal", "Preictal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{patient} — {model} ({split})\n"
                 f"AUC-ROC: {m['auc_roc']:.3f}  Recall: {m['recall']:.3f}")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{split}_{patient}_{model}.png", dpi=100)
    plt.close(fig)


def print_split_metrics(train_m: dict, val_m: dict, label: str):
    print(f"  [{label}]")
    print(f"    {'':8} {'AUC-ROC':>8} {'AUC-PR':>8} {'Loss':>8} "
          f"{'Recall':>8}  Confusion (TN  FP  FN  TP)")
    for tag, m in [("train", train_m), ("val", val_m)]:
        print(f"    {tag:<8} {m['auc_roc']:>8.3f} {m['auc_pr']:>8.3f} "
              f"{m['loss']:>8.4f} {m['recall']:>8.3f}  "
              f"[{m['tn']:>5} {m['fp']:>5} {m['fn']:>5} {m['tp']:>5}]")


def build_gb() -> GradientBoostingClassifier:
    """
    sklearn GradientBoosting — CPU only, does not use Metal GPU.
    Imbalance handled via compute_sample_weight('balanced').
    """
    return GradientBoostingClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )


def focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """
    alpha=0.75 boosts the rare preictal (positive) class.
    alpha_t is applied per-class so direction matches class_weight.
    """
    def loss(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return alpha_t * tf.pow(1.0 - p_t, gamma) * bce
    return loss


def build_nn(n_features: int) -> keras.Model:
    """
    Uses Metal GPU on Apple Silicon for faster training.
    Batch size 256 optimized for M4 unified memory.
    """
    inputs = keras.Input(shape=(n_features,))
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


# ─────────────────────────────────────────────────────────────
# LOAD MANIFEST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  MODEL TRAINING  v3 (Metal GPU optimized)")
print("=" * 62)

manifest_path = SPLIT_DIR / "manifest.json"
if not manifest_path.exists():
    print(f"Manifest not found at {manifest_path}")
    print("Run 05_split.py first.")
    exit(1)

with open(manifest_path) as f:
    manifest = json.load(f)

folds      = manifest["folds"]
n_features = manifest["n_features"]

print(f"  Folds      : {len(folds)}")
print(f"  Features   : {n_features}")
print(f"  Batch size : {BATCH_SIZE} (M4 optimized)")
print(f"  Models dir : {MODELS_DIR}/\n")


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
gb_metrics = {}
nn_metrics = {}

for fold_idx, patient in enumerate(folds):
    print(f"\n── Fold {fold_idx+1}/{len(folds)} "
          f"| Test patient: {patient} " + "─" * 20)

    fold_path = SPLIT_DIR / f"fold_{patient}.npz"
    if not fold_path.exists():
        print(f"  Fold file not found: {fold_path} — skipping")
        continue

    fold = np.load(str(fold_path))
    X_tr,  y_tr  = fold["X_tr"],  fold["y_tr"]
    X_val, y_val = fold["X_val"], fold["y_val"]

    n_pre_tr  = int((y_tr  == 1).sum())
    n_pre_val = int((y_val == 1).sum())
    print(f"  train: {len(y_tr):,}  "
          f"(pre: {n_pre_tr:,} | inter: {int((y_tr==0).sum()):,})")
    print(f"  val  : {len(y_val):,}  "
          f"(pre: {n_pre_val:,} | inter: {int((y_val==0).sum()):,})")

    # ── GradientBoosting ──────────────────────────────────
    print(f"\n  [GradientBoosting] training...")
    gb_model         = build_gb()
    sample_weight_tr = compute_sample_weight("balanced", y_tr)
    gb_model.fit(X_tr, y_tr, sample_weight=sample_weight_tr)

    gb_tr_m  = evaluate(y_tr,  gb_model.predict_proba(X_tr)[:, 1])
    gb_val_m = evaluate(y_val, gb_model.predict_proba(X_val)[:, 1])
    print_split_metrics(gb_tr_m, gb_val_m, "GradientBoosting")
    save_cm(gb_tr_m,  patient, "gb", "train")
    save_cm(gb_val_m, patient, "gb", "val")

    gb_metrics[patient] = {"train": gb_tr_m, "val": gb_val_m}
    joblib.dump(gb_model, MODELS_DIR / f"gb_{patient}.pkl")

    # ── Neural network with focal loss ────────────────────
    print(f"\n  [Neural net] training (Metal GPU)...")
    cw    = compute_class_weight("balanced",
                                  classes=np.array([0, 1]), y=y_tr)
    nn_cw = {0: float(cw[0]), 1: float(cw[1])}

    nn_model = build_nn(n_features)
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=7, restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=0,
        ),
    ]

    nn_model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=nn_cw,
        callbacks=callbacks,
        verbose=0,
    )

    nn_tr_m  = evaluate(y_tr,
                         nn_model.predict(X_tr,  verbose=0).flatten())
    nn_val_m = evaluate(y_val,
                         nn_model.predict(X_val, verbose=0).flatten())
    print_split_metrics(nn_tr_m, nn_val_m, "Neural net (Metal GPU)")
    save_cm(nn_tr_m,  patient, "nn", "train")
    save_cm(nn_val_m, patient, "nn", "val")

    nn_metrics[patient] = {"train": nn_tr_m, "val": nn_val_m}
    nn_model.save(str(MODELS_DIR / f"nn_{patient}.keras"))


# ─────────────────────────────────────────────────────────────
# SAVE + SUMMARIZE
# ─────────────────────────────────────────────────────────────
train_metrics_path = MODELS_DIR / "train_metrics.json"
with open(train_metrics_path, "w") as f:
    json.dump({"gradient_boosting": gb_metrics,
               "neural_net": nn_metrics}, f, indent=2)


def summarize_split(metrics: dict, model_name: str, split: str):
    rows = {p: m[split] for p, m in metrics.items() if split in m}
    if not rows:
        return
    aucs = [r["auc_roc"] for r in rows.values()]
    aprs = [r["auc_pr"]  for r in rows.values()]
    recs = [r["recall"]  for r in rows.values()]
    f1s  = [r["f1"]      for r in rows.values()]

    print(f"\n{'=' * 70}")
    print(f"  {model_name} — {split.upper()} SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Patient':<10} {'AUC-ROC':>8} {'AUC-PR':>8} "
          f"{'Recall':>8} {'F1':>7}  {'Loss':>8}")
    print("  " + "-" * 56)
    for pat, r in sorted(rows.items()):
        print(f"  {pat:<10} {r['auc_roc']:>8.3f} {r['auc_pr']:>8.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>7.3f}  {r['loss']:>8.4f}")
    print("  " + "-" * 56)
    print(f"  {'MEAN':<10} {np.mean(aucs):>8.3f} {np.mean(aprs):>8.3f} "
          f"{np.mean(recs):>8.3f} {np.mean(f1s):>7.3f}")
    print(f"  {'STD':<10} {np.std(aucs):>8.3f} {np.std(aprs):>8.3f} "
          f"{np.std(recs):>8.3f} {np.std(f1s):>7.3f}")
    print(f"{'=' * 70}")


summarize_split(gb_metrics, "GRADIENT BOOSTING", "train")
summarize_split(gb_metrics, "GRADIENT BOOSTING", "val")
summarize_split(nn_metrics, "NEURAL NETWORK",    "train")
summarize_split(nn_metrics, "NEURAL NETWORK",    "val")

print(f"\n  Train metrics saved → {train_metrics_path}")
print(f"  Confusion matrices  → {CM_DIR}/")
print(f"  Models saved        → {MODELS_DIR}/\n")