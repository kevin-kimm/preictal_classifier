"""
=============================================================
  Siena Scalp EEG — Validation Evaluation  v3
  07_evaluate.py

  Loads the held-out test split for each fold and evaluates
  both trained models. Reports per-patient and aggregate
  metrics with full confusion matrices.

  Reads : data/split/fold_{patient}.npz
          data/split/manifest.json
          models_v3/gb_{patient}.pkl
          models_v3/nn_{patient}.keras
  Output: models_v3/eval_results.json
=============================================================
"""

import json
import joblib
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    f1_score, precision_score, recall_score, confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

SPLIT_DIR  = Path("data/split")
MODELS_DIR = Path("models_v3")
THRESHOLD  = 0.65
CM_DIR = MODELS_DIR / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)


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
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{split}_{patient}_{model}.png", dpi=100)
    plt.close(fig)


def print_metrics(m: dict, label: str):
    print(f"  {label:<18}  AUC-ROC: {m['auc_roc']:.3f}  AUC-PR: {m['auc_pr']:.3f}  "
          f"Loss: {m['loss']:.4f}  Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}")
    print(f"  {'':18}  Confusion → "
          f"TN: {m['tn']:>5}  FP: {m['fp']:>5}  FN: {m['fn']:>5}  TP: {m['tp']:>5}")


# Load manifest
print("\n" + "=" * 62)
print("  EVALUATION ON HELD-OUT TEST  v3")
print("=" * 62)

manifest_path = SPLIT_DIR / "manifest.json"
if not manifest_path.exists():
    print(f"Manifest not found at {manifest_path}")
    print("Run 05_split.py first.")
    exit(1)

with open(manifest_path) as f:
    manifest = json.load(f)

folds = manifest["folds"]
print(f"  Folds: {len(folds)}\n")


# Evaluation loop
gb_results = {}
nn_results = {}

for fold_idx, patient in enumerate(folds):
    print(f"── Fold {fold_idx+1}/{len(folds)} | Patient: {patient} " + "─" * 25)

    fold_path = SPLIT_DIR / f"fold_{patient}.npz"
    gb_path   = MODELS_DIR / f"gb_{patient}.pkl"
    nn_path   = MODELS_DIR / f"nn_{patient}.keras"

    missing = [p for p in [fold_path, gb_path, nn_path] if not p.exists()]
    if missing:
        print(f"  Missing files: {[str(p) for p in missing]} — skipping\n")
        continue

    fold   = np.load(str(fold_path))
    X_test = fold["X_test"]
    y_test = fold["y_test"]

    print(f"  Test: {len(y_test):,} windows  "
          f"(pre: {int((y_test==1).sum()):,} | inter: {int((y_test==0).sum()):,})")

    # GradientBoosting
    gb_model = joblib.load(gb_path)
    gb_prob  = gb_model.predict_proba(X_test)[:, 1]
    gb_m     = evaluate(y_test, gb_prob)
    gb_results[patient] = gb_m
    print_metrics(gb_m, "GradientBoosting")
    save_cm(gb_m, patient, "gb", "test")

    # Neural network
    nn_model = keras.models.load_model(
        str(nn_path),
        custom_objects={"loss": lambda y, p: p},  # placeholder; focal loss baked in
        compile=False,
    )
    nn_prob = nn_model.predict(X_test, verbose=0).flatten()
    nn_m    = evaluate(y_test, nn_prob)
    nn_results[patient] = nn_m
    print_metrics(nn_m, "Neural net")
    save_cm(nn_m, patient, "nn", "test")
    print()


# Save results
eval_path = MODELS_DIR / "eval_results.json"
with open(eval_path, "w") as f:
    json.dump({"gradient_boosting": gb_results, "neural_net": nn_results}, f, indent=2)


# Summary tables
def summarize(results: dict, model_name: str):
    if not results:
        return
    aucs = [r["auc_roc"] for r in results.values()]
    aprs = [r["auc_pr"]  for r in results.values()]
    recs = [r["recall"]  for r in results.values()]
    f1s  = [r["f1"]      for r in results.values()]
    prec = [r["precision"] for r in results.values()]

    print(f"\n{'=' * 76}")
    print(f"  {model_name} — TEST SUMMARY (held-out patients)")
    print(f"{'=' * 76}")
    print(f"  {'Patient':<10} {'AUC-ROC':>8} {'AUC-PR':>8} {'Recall':>8} "
          f"{'Precision':>10} {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 72)
    for pat, r in sorted(results.items()):
        print(f"  {pat:<10} {r['auc_roc']:>8.3f} {r['auc_pr']:>8.3f} "
              f"{r['recall']:>8.3f} {r['precision']:>10.3f} {r['f1']:>7.3f}  "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")
    print("  " + "-" * 72)
    print(f"  {'MEAN':<10} {np.mean(aucs):>8.3f} {np.mean(aprs):>8.3f} "
          f"{np.mean(recs):>8.3f} {np.mean(prec):>10.3f} {np.mean(f1s):>7.3f}")
    print(f"  {'STD':<10} {np.std(aucs):>8.3f} {np.std(aprs):>8.3f} "
          f"{np.std(recs):>8.3f} {np.std(prec):>10.3f} {np.std(f1s):>7.3f}")
    print(f"{'=' * 76}")


summarize(gb_results, "GRADIENT BOOSTING")
summarize(nn_results, "NEURAL NETWORK")


# Head-to-head
common = sorted(set(gb_results) & set(nn_results))
if common:
    print(f"\n{'=' * 62}")
    print(f"  Head-to-head: GradientBoosting vs Neural Network (AUC-ROC)")
    print(f"{'=' * 62}")
    print(f"  {'Patient':<10} {'GB AUC':>9} {'NN AUC':>9} {'Winner':>14}")
    print("  " + "-" * 46)
    gb_wins = nn_wins = 0
    for pat in common:
        ga = gb_results[pat]["auc_roc"]
        na = nn_results[pat]["auc_roc"]
        if ga > na:
            gb_wins += 1
            winner = "GradBoost"
        else:
            nn_wins += 1
            winner = "Neural net"
        print(f"  {pat:<10} {ga:>9.3f} {na:>9.3f} {winner:>14}")
    print("  " + "-" * 46)
    print(f"  GradBoost wins: {gb_wins}  |  Neural net wins: {nn_wins}")
    print(f"{'=' * 62}")

print(f"\n  Results saved        → {eval_path}")
print(f"  Confusion matrices   → {CM_DIR}/\n")
