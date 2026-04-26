"""
=============================================================
  Siena Scalp EEG — Patient-Specific Fine-Tuning v6
  07_finetune.py

  Fine-tunes the final 2 layers of each patient's best
  cross-patient model on that patient's own data.

  Simulates real deployment:
    Device ships with cross-patient model
    Patient records 1-2 seizures (calibration period)
    Final layers fine-tuned on patient's own brain data
    Accuracy improves significantly for that patient

  Reads  : data/features/features_v6.npz
           models_v6/nn_PN*.keras
  Output : models_v6/nn_PN*_finetuned.keras
           models_v6/finetuned_results.json
=============================================================
"""

import os
import json
import numpy as np
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)


# Configuration 
FEATURES_PATH   = Path("data/features/features_v6.npz")
MODELS_DIR      = Path("models_v6")
THRESHOLD       = 0.65
STEP_SEC        = 5

# CRITICAL: very small learning rate for fine tuning
# We don't want to "forget" what the model learned from
# 13 patients — we just want to nudge the final decision
# boundary toward this specific patient's brain patterns
FINETUNE_LR     = 1e-5    # 0.00001 — 100x smaller than training

FINETUNE_EPOCHS = 50      # short — don't overfit to limited data

# Use 70% of patient's data for fine-tuning
# Keep 30% as held-out evaluation (never seen during fine-tuning)
FINETUNE_SPLIT  = 0.70


# Metrics function
def evaluate(y_true, y_prob, threshold=THRESHOLD):
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
    }


# FINE-TUNING 
# The cross-patient model has 4 layers:
#   Layer 1 (Dense 128) : learned general EEG patterns
#                         from 13 different patients
#                         → FREEZE this (universal knowledge)
#   Layer 2 (Dense 64)  : learned combinations of patterns
#                         → FREEZE this (universal knowledge)
#   Layer 3 (Dense 32)  : higher-level abstractions
#                         → UNFREEZE: adapt to this patient
#   Layer 4 (Dense 1)   : final decision boundary
#                         → UNFREEZE: adapt to this patient
#
# By freezing early layers, we keep the general EEG knowledge.
# By retraining final layers, we adapt the decision to this brain.
# This is called "transfer learning" — used everywhere in AI.
# Medical analogy: a doctor learns medicine (cross-patient),
# then specializes in one patient's condition (fine-tuning).


print("\n" + "=" * 66)
print("  PATIENT-SPECIFIC FINE-TUNING v6")
print("=" * 66)
print(f"""
  What this does:
    Takes the best cross-patient model per patient
    Fine-tunes ONLY the final 2 layers on patient's own data
    Early layers (general EEG knowledge) are FROZEN

  Why: Every brain is different. The cross-patient model
    learned general preictal patterns. Fine-tuning adapts
    the final decision boundary to THIS patient's specific
    preictal signature.

  Settings:
    LR     : {FINETUNE_LR}  (tiny — don't forget cross-patient knowledge)
    Epochs : {FINETUNE_EPOCHS}
    Split  : {int(FINETUNE_SPLIT*100)}% fine-tune / {int((1-FINETUNE_SPLIT)*100)}% evaluate
""")


# LOAD DATA
data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]
patient_ids = sorted(set(patients))

base_results = {}   # performance BEFORE fine-tuning
ft_results   = {}   # performance AFTER fine-tuning



# FINE-TUNING LOOP — one patient at a time
for patient in patient_ids:
    # Get all data for this specific patient
    mask  = patients == patient
    X_pat = X_all[mask]
    y_pat = y_all[mask]
    n_pre = int((y_pat == 1).sum())

    # Need at least 10 preictal windows to fine-tune meaningfully
    if n_pre < 10:
        print(f"  {patient} — only {n_pre} preictal — skipping")
        continue

    nn_path = MODELS_DIR / f"nn_{patient}.keras"
    if not nn_path.exists():
        print(f"  {patient} — model not found — skipping")
        continue

    # Stratified Split 
    # Split patient's data while maintaining the preictal ratio
    # Important: don't accidentally put all preictal in one split
    pre_idx   = np.where(y_pat == 1)[0]   # indices of preictal windows
    inter_idx = np.where(y_pat == 0)[0]   # indices of interictal windows

    # 70% of each class for fine-tuning
    n_pt = max(5, int(len(pre_idx)   * FINETUNE_SPLIT))
    n_it = max(5, int(len(inter_idx) * FINETUNE_SPLIT))

    # Combine indices for fine-tune set and eval set
    tune_idx = np.concatenate([pre_idx[:n_pt], inter_idx[:n_it]])
    eval_idx = np.concatenate([pre_idx[n_pt:], inter_idx[n_it:]])

    X_tune = X_pat[tune_idx]; y_tune = y_pat[tune_idx]
    X_eval = X_pat[eval_idx]; y_eval = y_pat[eval_idx]

    if (y_eval == 1).sum() == 0:
        print(f"  {patient} — no preictal in eval — skipping")
        continue

    # Scale features using fine-tune data
    # (in deployment, this scaler would be fitted on calibration data)
    scaler        = StandardScaler()
    X_tune_scaled = scaler.fit_transform(X_tune)
    X_eval_scaled = scaler.transform(X_eval)

    n_pt_n = int((y_tune == 1).sum())   # preictal count in tune set
    n_it_n = int((y_tune == 0).sum())   # interictal count in tune set
    spw    = n_it_n / max(n_pt_n, 1)   # imbalance ratio for class weights

    # Evaluate BASE model before any fine-tuning 
    # This is the honest "off the shelf" performance
    base = keras.models.load_model(str(nn_path), compile=False)
    base_prob = base.predict(X_eval_scaled, verbose=0).flatten()
    base_m    = evaluate(y_eval, base_prob)
    base_results[patient] = base_m

    # Build fine-tuned model 
    # Load the same weights again (fresh copy to fine-tune)
    ft = keras.models.load_model(str(nn_path), compile=False)

    # FREEZE all layers — stop all weight updates
    for layer in ft.layers:
        layer.trainable = False

    # UNFREEZE only the last 2 layers
    # These are: Dense(32) and Dense(1) output
    # They form the final "decision" part of the network
    for layer in ft.layers[-2:]:
        layer.trainable = True

    # Compile with tiny learning rate
    # Using same loss as training for consistency
    ft.compile(
        optimizer=keras.optimizers.Adam(FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    # Fine-tune on patient's own data
    # Small batch size because we have limited patient data
    ft.fit(
        X_tune_scaled, y_tune,
        batch_size=min(32, max(8, len(y_tune)//4)),
        epochs=FINETUNE_EPOCHS,
        class_weight={0: 1.0, 1: spw},
        callbacks=[
            # Stop if training AUC stops improving
            # (prevents overfitting to small dataset)
            keras.callbacks.EarlyStopping(
                monitor="auc", mode="max",
                patience=10, restore_best_weights=True,
                verbose=0)],
        verbose=0,
    )

    # Evaluate FINE-TUNED model 
    ft_prob = ft.predict(X_eval_scaled, verbose=0).flatten()
    ft_m    = evaluate(y_eval, ft_prob)
    ft_results[patient] = ft_m

    # Calculate and display improvement
    gain  = ft_m["auc_roc"] - base_m["auc_roc"]
    arrow = "↑" if gain > 0.01 else "↓" if gain < -0.01 else "→"
    grd   = "✅" if ft_m["auc_roc"] >= 0.70 else \
            "⚠️ " if ft_m["auc_roc"] >= 0.60 else "❌"
    print(f"  {patient}  "
          f"tune:{n_pt_n}pre  eval:{int((y_eval==1).sum())}pre  "
          f"base:{base_m['auc_roc']:.3f} -> "
          f"fine:{ft_m['auc_roc']:.3f}  "
          f"{arrow}{gain:+.3f}  {grd}")

    # Save fine-tuned model with _finetuned suffix
    # Keeps original model intact (base) and adds fine-tuned version
    ft.save(str(MODELS_DIR / f"nn_{patient}_finetuned.keras"))


# SUMMARY — Before vs After fine-tuning
print(f"\n{'=' * 66}")
print(f"  FINE-TUNING SUMMARY")
print(f"{'=' * 66}")
print(f"  {'Patient':<10} {'Base':>8} {'FT':>8} "
      f"{'Gain':>7} {'FP/10':>7}  Grade")
print("  " + "-" * 50)

base_aucs, ft_aucs = [], []
for pat in sorted(ft_results.keys()):
    ba = base_results[pat]["auc_roc"]   # before fine-tuning
    fa = ft_results[pat]["auc_roc"]     # after fine-tuning
    gain = fa - ba
    base_aucs.append(ba)
    ft_aucs.append(fa)
    arrow = "↑" if gain > 0.01 else "↓" if gain < -0.01 else "→"
    grd   = "✅" if fa >= 0.70 else "⚠️ " if fa >= 0.60 else "❌"
    print(f"  {pat:<10} {ba:>8.3f} {fa:>8.3f} "
          f"{gain:>+7.3f} "
          f"{ft_results[pat]['fp_per_10']:>7}  {arrow} {grd}")

print("  " + "-" * 50)
if base_aucs:
    print(f"  {'MEAN':<10} {np.mean(base_aucs):>8.3f} "
          f"{np.mean(ft_aucs):>8.3f} "
          f"{np.mean(ft_aucs)-np.mean(base_aucs):>+7.3f}")

pb = sum(1 for a in base_aucs if a >= 0.70)
pf = sum(1 for a in ft_aucs   if a >= 0.70)
print(f"\n  Before fine-tuning : {pb}/{len(base_aucs)} predictable")
print(f"  After  fine-tuning : {pf}/{len(ft_aucs)} predictable "
      f"(+{pf-pb})")
print(f"\n  This demonstrates the clinical deployment path:")
print(f"  1. Device ships with cross-patient model (general)")
print(f"  2. Patient wears device — 1-2 seizures recorded")
print(f"  3. Fine-tune final 2 layers on patient's own data")
print(f"  4. Personal model activated — higher accuracy")


# Save results for 08_final_eval.py
with open(MODELS_DIR / "finetuned_results.json", "w") as f:
    json.dump({"base":      base_results,
               "finetuned": ft_results,
               "summary": {
                   "pred_before": pb,
                   "pred_after":  pf,
                   "mean_before": float(np.mean(base_aucs)),
                   "mean_after":  float(np.mean(ft_aucs)),
               }}, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/finetuned_results.json\n")