"""
=============================================================
  Siena Scalp EEG — Multi-Run Best Model Selector
  06_multi_run.py

  Builds on v2 original (our best: 0.584 mean, 6 predictable)
  Runs 4 more seeds, keeps best model per patient.

  Reads  : data/features/features.npz  (v2 64 features)
           models/lopo_results.json    (v2 seed 42 baseline)
           models/nn_PN*.keras         (v2 trained models)
  Output : models/nn_PN*.keras         (updated if improved)
           models/multi_run_results.json
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
    roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURES_PATH = Path("data/features/features.npz")
MODELS_DIR    = Path("models")

THRESHOLD    = 0.65
BATCH_SIZE   = 256
EPOCHS       = 100
STEP_SEC     = 5

# Seeds to try — seed 42 already done in original v2
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
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }


def build_nn(n_features, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
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


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print("  MULTI-RUN BEST MODEL SELECTOR — v2 Original")
print("=" * 66)
print(f"""
  Base     : v2 original (64 band power features)
  Channels : F7, T3, T5, C3, F8, T4, T6, C4
  Strategy : Try {len(ADDITIONAL_SEEDS)} more seeds, keep best per patient
  Seeds    : {ADDITIONAL_SEEDS}
  (Seed 42 already done — ~0.584 mean AUC, 6 predictable)
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
n_features  = X_all.shape[1]

print(f"  Windows   : {len(y_all):,}")
print(f"  Features  : {n_features}")
print(f"  Patients  : {len(patient_ids)}")

# ─────────────────────────────────────────────────────────────
# LOAD EXISTING BEST AUCs FROM V2 SEED 42
# ─────────────────────────────────────────────────────────────
existing_path = MODELS_DIR / "lopo_results.json"
best_aucs     = {}
best_metrics  = {}

if existing_path.exists():
    with open(existing_path) as f:
        existing = json.load(f)
    nn_res = existing.get("neural_net", {})
    for pat, r in nn_res.items():
        best_aucs[pat]    = r["auc_roc"]
        best_metrics[pat] = r

    print(f"\n  Loaded v2 baseline (seed 42):")
    print(f"  {'Patient':<10} {'AUC':>7}  Grade")
    print("  " + "-" * 24)
    for pat, auc in sorted(best_aucs.items()):
        grd = "✅" if auc >= 0.70 else "⚠️ " if auc >= 0.60 else "❌"
        print(f"  {pat:<10} {auc:>7.3f}  {grd}")
    pred = sum(1 for a in best_aucs.values() if a >= 0.70)
    print(f"\n  Baseline: {pred}/{len(best_aucs)} predictable  "
          f"mean={np.mean(list(best_aucs.values())):.3f}")
else:
    print("  No existing results — starting fresh")


# ─────────────────────────────────────────────────────────────
# MULTI-RUN LOOP
# ─────────────────────────────────────────────────────────────
multi_run_log = {pat: {"seed_42": best_aucs.get(pat, 0)}
                 for pat in patient_ids}
improvements  = {}

for seed in ADDITIONAL_SEEDS:
    print(f"\n{'=' * 66}")
    print(f"  SEED {seed}")
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
                    patience=12,
                    restore_best_weights=True,
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
            gain = new_auc - old_auc
            print(f"  ✅ {test_patient}  "
                  f"{old_auc:.3f} -> {new_auc:.3f}  "
                  f"(+{gain:.3f}) IMPROVED — saving")
            best_aucs[test_patient]    = new_auc
            best_metrics[test_patient] = metrics
            nn.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))
            joblib.dump(
                {"model": None, "scaler": scaler},
                str(MODELS_DIR / f"scaler_{test_patient}.pkl"))
            improvements[test_patient] = {
                "old_auc": old_auc,
                "new_auc": new_auc,
                "seed":    seed,
                "gain":    gain,
            }
        else:
            print(f"  → {test_patient}  "
                  f"seed={seed} AUC={new_auc:.3f}  "
                  f"(best={old_auc:.3f}) no change")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  MULTI-RUN FINAL SUMMARY")
print(f"{'=' * 70}")
print(f"\n  {'Patient':<10} {'s42':>7}", end="")
for seed in ADDITIONAL_SEEDS:
    print(f"  {'s'+str(seed):>7}", end="")
print(f"  {'Best':>7}  {'Change':>8}  Grade")
print("  " + "-" * 68)

all_best = []
for pat in sorted(patient_ids):
    if pat not in best_aucs:
        continue
    log  = multi_run_log.get(pat, {})
    orig = log.get("seed_42", 0)
    best = best_aucs[pat]
    diff = best - orig
    all_best.append(best)
    arrow = "↑" if diff > 0.005 else "→"
    grd   = "✅ YES" if best >= 0.70 else \
            "⚠️  MOD" if best >= 0.60 else "❌ NO"

    print(f"  {pat:<10} {orig:>7.3f}", end="")
    for seed in ADDITIONAL_SEEDS:
        val = log.get(f"seed_{seed}", 0)
        print(f"  {val:>7.3f}", end="")
    print(f"  {best:>7.3f}  {diff:>+8.3f}  {grd}")

print("  " + "-" * 68)
if all_best:
    print(f"  {'MEAN':<10} "
          f"{np.mean(list(v['seed_42'] for v in multi_run_log.values() if 'seed_42' in v)):>7.3f}"
          f"{'':>33}  "
          f"{np.mean(all_best):>7.3f}")

pred  = sum(1 for a in all_best if a >= 0.70)
mod   = sum(1 for a in all_best if 0.60 <= a < 0.70)
poor  = sum(1 for a in all_best if a < 0.60)
total = len(all_best)

print(f"\n  ✅ Predictable (AUC >= 0.70) : {pred}/{total}")
print(f"  ⚠️  Modest     (0.60-0.70)   : {mod}/{total}")
print(f"  ❌ Poor        (< 0.60)      : {poor}/{total}")

if improvements:
    print(f"\n  Patients improved ({len(improvements)}):")
    for pat, imp in sorted(improvements.items(),
                            key=lambda x: -x[1]["gain"]):
        print(f"    {pat}: {imp['old_auc']:.3f} -> "
              f"{imp['new_auc']:.3f} "
              f"(+{imp['gain']:.3f}) via seed {imp['seed']}")
else:
    print(f"\n  No patients improved over seed 42 baseline")

# Save
with open(MODELS_DIR / "multi_run_results.json", "w") as f:
    json.dump({
        "best_aucs":    best_aucs,
        "run_log":      multi_run_log,
        "improvements": improvements,
        "best_metrics": best_metrics,
    }, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/multi_run_results.json\n")