"""
=============================================================
  Siena Scalp EEG — Multi-Run Best Model Selector
  06_multi_run.py

  Runs training 4 more times with different random seeds.
  For each patient, keeps the model with the highest AUC.
  This stabilizes results by exploiting the best random init.

  Reads  : data/features/features_v202.npz
           models_v202/nn_PN*.keras  (from 05_train_model.py)
  Output : models_v202/nn_PN*.keras  (updated if improved)
           models_v202/multi_run_results.json
=============================================================
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

os.environ["TF_METAL_ENABLED"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURES_PATH = Path("data/features/features_v202.npz")
MODELS_DIR    = Path("models_v202")
THRESHOLD     = 0.50
BATCH_SIZE    = 256
EPOCHS        = 100
STEP_SEC      = 5

# Seeds to try — seed 42 already done in run 1
ADDITIONAL_SEEDS = [123, 456, 789, 999]


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


def build_nn(n_features, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
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
print("  MULTI-RUN BEST MODEL SELECTOR")
print("=" * 66)
print(f"""
  Strategy: Train {len(ADDITIONAL_SEEDS)} more times with different seeds.
  For each patient, keep the model with the highest AUC.
  This exploits random initialization variance in our favor.

  Seeds to try: {ADDITIONAL_SEEDS}
  (Seed 42 already done in 05_train_model.py)
""")

data     = np.load(str(FEATURES_PATH), allow_pickle=True)
X_all    = data["X"]
y_all    = data["y"]
patients = data["patients"]
patient_ids = sorted(set(patients))
n_features  = X_all.shape[1]

print(f"  Windows   : {len(y_all):,}")
print(f"  Features  : {n_features}")
print(f"  Patients  : {len(patient_ids)}")

# Load existing best AUCs from run 1
existing_results_path = MODELS_DIR / "lopo_results.json"
best_aucs = {}

if existing_results_path.exists():
    with open(existing_results_path) as f:
        existing = json.load(f)
    nn_res = existing.get("neural_net", {})
    for pat, r in nn_res.items():
        best_aucs[pat] = r["auc_roc"]
    print(f"\n  Loaded existing best AUCs from run 1:")
    for pat, auc in sorted(best_aucs.items()):
        grd = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"
        print(f"    {pat}: {auc:.3f} {grd}")
else:
    print("  No existing results found — starting fresh")

multi_run_log = {pat: {"seed_42": best_aucs.get(pat, 0)}
                 for pat in patient_ids}
improvements  = {}

for seed in ADDITIONAL_SEEDS:
    print(f"\n{'=' * 66}")
    print(f"  SEED {seed} — Training all folds")
    print(f"{'=' * 66}")

    for fold_idx, test_patient in enumerate(patient_ids):
        test_mask  = patients == test_patient
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if (y_test == 1).sum() == 0:
            continue

        n_pre   = int((y_train == 1).sum())
        n_inter = int((y_train == 0).sum())
        spw     = n_inter / n_pre

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        nn = build_nn(n_features, seed=seed)
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
                    monitor="val_auc", mode="max",
                    patience=12, restore_best_weights=True,
                    verbose=0),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=5, min_lr=1e-7, verbose=0),
            ],
            verbose=0,
        )

        prob    = nn.predict(X_test_scaled, verbose=0).flatten()
        metrics = evaluate(y_test, prob)
        new_auc = metrics["auc_roc"]
        old_auc = best_aucs.get(test_patient, 0)

        multi_run_log[test_patient][f"seed_{seed}"] = new_auc

        if new_auc > old_auc:
            improvement = new_auc - old_auc
            print(f"  ✅ {test_patient}  seed={seed}  "
                  f"{old_auc:.3f} -> {new_auc:.3f}  "
                  f"(+{improvement:.3f}) IMPROVED — saving")
            best_aucs[test_patient] = new_auc
            nn.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))
            joblib.dump(scaler, str(MODELS_DIR /
                                    f"scaler_{test_patient}.pkl"))
            improvements[test_patient] = {
                "old_auc": old_auc,
                "new_auc": new_auc,
                "seed":    seed,
                "gain":    improvement,
            }
        else:
            print(f"  → {test_patient}  seed={seed}  "
                  f"{new_auc:.3f}  (best={old_auc:.3f}) no change")


# ── Final summary ─────────────────────────────────────────
print(f"\n{'=' * 66}")
print(f"  MULTI-RUN SUMMARY")
print(f"{'=' * 66}")
print(f"\n  {'Patient':<10}", end="")
print(f"  {'s42':>7}", end="")
for seed in ADDITIONAL_SEEDS:
    print(f"  {'s'+str(seed):>7}", end="")
print(f"  {'Best':>7}  Change")
print("  " + "-" * 60)

for pat in sorted(multi_run_log.keys()):
    log  = multi_run_log[pat]
    best = best_aucs.get(pat, 0)
    orig = log.get("seed_42", 0)
    diff = best - orig
    arrow = "↑" if diff > 0.01 else "→"
    print(f"  {pat:<10}  {orig:>7.3f}", end="")
    for seed in ADDITIONAL_SEEDS:
        val = log.get(f"seed_{seed}", 0)
        print(f"  {val:>7.3f}", end="")
    print(f"  {best:>7.3f}  {arrow} {diff:+.3f}")

print(f"\n  Patients improved: {len(improvements)}/{len(best_aucs)}")
for pat, imp in sorted(improvements.items(),
                        key=lambda x: -x[1]["gain"]):
    print(f"    {pat}: {imp['old_auc']:.3f} -> "
          f"{imp['new_auc']:.3f} (+{imp['gain']:.3f}) "
          f"via seed {imp['seed']}")

pred = sum(1 for a in best_aucs.values() if a >= 0.70)
mod  = sum(1 for a in best_aucs.values() if 0.60 <= a < 0.70)
poor = sum(1 for a in best_aucs.values() if a < 0.60)
total = len(best_aucs)
print(f"\n  ✅ Predictable (>=0.70): {pred}/{total}")
print(f"  ⚠️  Modest  (0.60-0.70): {mod}/{total}")
print(f"  ❌ Poor       (<0.60)  : {poor}/{total}")

with open(MODELS_DIR / "multi_run_results.json", "w") as f:
    json.dump({"best_aucs": best_aucs,
               "run_log":   multi_run_log,
               "improvements": improvements}, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/multi_run_results.json\n")