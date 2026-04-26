"""
=============================================================
  Siena Scalp EEG — Patient-Specific Fine-Tuning
  07_finetune.py

  Takes the best cross-patient model per patient and
  fine-tunes it on that patient's own data.

  This simulates real deployment:
    1. Patient wears device for calibration period
    2. 1-2 seizures are recorded with labels
    3. Model fine-tunes final layers on patient's data
    4. AUC improves significantly

  Strategy:
    - Freeze all layers except the last 2
    - Fine-tune with tiny learning rate (1e-5)
    - Use only that patient's own preictal + interictal data
    - Evaluate on held-out portion of patient's data

  This is how ALL commercial seizure devices work.
  Output: models_v202/finetuned_results.json
=============================================================
"""

import os
import json
import numpy as np
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow import keras

FEATURES_PATH = Path("data/features/features_v202.npz")
MODELS_DIR    = Path("models_v202")
THRESHOLD     = 0.50
STEP_SEC      = 5
FINETUNE_LR   = 1e-5   # very small — don't forget cross-patient knowledge
FINETUNE_EPOCHS = 30
# Use 70% of patient data for fine-tuning, 30% for evaluation
FINETUNE_SPLIT  = 0.70


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


print("\n" + "=" * 66)
print("  PATIENT-SPECIFIC FINE-TUNING")
print("=" * 66)
print(f"""
  Simulates real deployment:
    Phase 1 — Cross-patient model (already trained)
    Phase 2 — Patient wears device, 1-2 seizures recorded
    Phase 3 — Fine-tune final layers on patient's own data
    Phase 4 — Personal model deployed on device

  Fine-tune settings:
    Learning rate : {FINETUNE_LR}  (tiny — preserve cross-patient knowledge)
    Epochs        : {FINETUNE_EPOCHS}
    Split         : {int(FINETUNE_SPLIT*100)}% fine-tune / {int((1-FINETUNE_SPLIT)*100)}% evaluate
    Frozen layers : all except last 2
""")

data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]
patient_ids = sorted(set(patients))

base_results     = {}
finetune_results = {}

for patient in patient_ids:
    patient_mask = patients == patient
    X_pat = X_all[patient_mask]
    y_pat = y_all[patient_mask]

    n_pre = int((y_pat == 1).sum())
    if n_pre < 10:
        print(f"  {patient} — only {n_pre} preictal windows, skipping")
        continue

    nn_path = MODELS_DIR / f"nn_{patient}.keras"
    if not nn_path.exists():
        print(f"  {patient} — model not found, skipping")
        continue

    # Split patient data: 70% fine-tune, 30% evaluate
    n_total   = len(y_pat)
    n_tune    = int(n_total * FINETUNE_SPLIT)

    # Stratified split — keep ratio of pre/inter
    pre_idx   = np.where(y_pat == 1)[0]
    inter_idx = np.where(y_pat == 0)[0]

    n_pre_tune   = max(1, int(len(pre_idx)   * FINETUNE_SPLIT))
    n_inter_tune = max(1, int(len(inter_idx) * FINETUNE_SPLIT))

    tune_idx = np.concatenate([
        pre_idx[:n_pre_tune],
        inter_idx[:n_inter_tune]
    ])
    eval_idx = np.concatenate([
        pre_idx[n_pre_tune:],
        inter_idx[n_inter_tune:]
    ])

    X_tune = X_pat[tune_idx]
    y_tune = y_pat[tune_idx]
    X_eval = X_pat[eval_idx]
    y_eval = y_pat[eval_idx]

    if (y_eval == 1).sum() == 0:
        print(f"  {patient} — no preictal in eval split, skipping")
        continue

    # Scale using fine-tune data
    scaler         = StandardScaler()
    X_tune_scaled  = scaler.fit_transform(X_tune)
    X_eval_scaled  = scaler.transform(X_eval)

    # Load base model and evaluate before fine-tuning
    base_model = keras.models.load_model(
        str(nn_path), compile=False)
    base_prob  = base_model.predict(X_eval_scaled,
                                     verbose=0).flatten()
    base_m     = evaluate(y_eval, base_prob)
    base_results[patient] = base_m

    # Fine-tune — freeze all layers except last 2
    ft_model = keras.models.load_model(
        str(nn_path), compile=False)

    # Freeze all layers
    for layer in ft_model.layers:
        layer.trainable = False

    # Unfreeze last 2 layers (dense32 and output)
    for layer in ft_model.layers[-2:]:
        layer.trainable = True

    n_pre_tune_actual   = int((y_tune == 1).sum())
    n_inter_tune_actual = int((y_tune == 0).sum())
    spw = n_inter_tune_actual / max(n_pre_tune_actual, 1)

    ft_model.compile(
        optimizer=keras.optimizers.Adam(FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    ft_model.fit(
        X_tune_scaled, y_tune,
        batch_size=32,
        epochs=FINETUNE_EPOCHS,
        class_weight={0: 1.0, 1: spw},
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="auc", mode="max",
                patience=8, restore_best_weights=True,
                verbose=0),
        ],
        verbose=0,
    )

    ft_prob = ft_model.predict(X_eval_scaled,
                                verbose=0).flatten()
    ft_m    = evaluate(y_eval, ft_prob)
    finetune_results[patient] = ft_m

    gain = ft_m["auc_roc"] - base_m["auc_roc"]
    arrow = "↑" if gain > 0.01 else "↓" if gain < -0.01 else "→"

    print(f"  {patient}  "
          f"tune:{n_pre_tune_actual}pre/{n_inter_tune_actual}inter  "
          f"eval:{int((y_eval==1).sum())}pre  "
          f"base:{base_m['auc_roc']:.3f} -> "
          f"fine:{ft_m['auc_roc']:.3f}  "
          f"{arrow} {gain:+.3f}")

    ft_model.save(str(MODELS_DIR /
                      f"nn_{patient}_finetuned.keras"))


# ── Summary ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  FINE-TUNING RESULTS")
print(f"{'=' * 70}")
print(f"  {'Patient':<10} {'Base AUC':>9} {'Fine AUC':>9} "
      f"{'Gain':>7}  Grade (after fine-tune)")
print("  " + "-" * 52)

base_aucs = []
ft_aucs   = []

for pat in sorted(finetune_results.keys()):
    base_auc = base_results[pat]["auc_roc"]
    ft_auc   = finetune_results[pat]["auc_roc"]
    gain     = ft_auc - base_auc
    base_aucs.append(base_auc)
    ft_aucs.append(ft_auc)
    arrow = "↑" if gain > 0.01 else "↓" if gain < -0.01 else "→"
    grd   = "✅" if ft_auc >= 0.70 else \
            "⚠️ " if ft_auc >= 0.60 else "❌"
    print(f"  {pat:<10} {base_auc:>9.3f} {ft_auc:>9.3f} "
          f"{gain:>+7.3f}  {arrow} {grd}")

print("  " + "-" * 52)
if base_aucs:
    print(f"  {'MEAN':<10} {np.mean(base_aucs):>9.3f} "
          f"{np.mean(ft_aucs):>9.3f} "
          f"{np.mean(ft_aucs)-np.mean(base_aucs):>+7.3f}")

pred_base = sum(1 for a in base_aucs if a >= 0.70)
pred_ft   = sum(1 for a in ft_aucs   if a >= 0.70)
print(f"\n  Before fine-tuning: {pred_base}/{len(base_aucs)} predictable")
print(f"  After  fine-tuning: {pred_ft}/{len(ft_aucs)} predictable")
print(f"\n  This simulates a patient wearing the device for a")
print(f"  calibration period. In real deployment, fine-tuning")
print(f"  would use data from multiple seizures over weeks.")

with open(MODELS_DIR / "finetuned_results.json", "w") as f:
    json.dump({"base":      base_results,
               "finetuned": finetune_results}, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/finetuned_results.json\n")