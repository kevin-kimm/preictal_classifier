"""
=============================================================
  Siena Scalp EEG — Model Training v6
  05_train_model.py

  Clean optimized training — seed 42 baseline run.
  This is Run 1. Run 06_multi_run.py after for best results.

  Config:
    Channels  : F7, T3, T5, C3, F8, T4, T6, C4
    Features  : 64 band powers
    Loss      : Binary crossentropy + class weights
    Arch      : 128->64->32->1
    Threshold : 0.65
    Batch     : 256 (M4 Metal GPU optimized)
    Epochs    : 100, patience=12

  Output: models_v6/lopo_results.json
          models_v6/nn_PN*.keras
          models_v6/scaler_PN*.pkl
=============================================================
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

# Tell TensorFlow to use Metal GPU on Apple Silicon 
# Must be set before importing tensorflow
os.environ["TF_METAL_ENABLED"] = "1"

# ML imports 
# StandardScaler : normalizes features to mean=0, std=1
#                  fitted on TRAINING data only, applied to test
#                  prevents data leakage from test set
from sklearn.preprocessing import StandardScaler

# Metrics for evaluating model performance:
# roc_auc_score   : AUC-ROC — main metric, threshold-independent
# confusion_matrix : TP, FP, FN, TN counts
# f1_score        : harmonic mean of precision and recall
# precision_score : of all alerts fired, % that were correct
# recall_score    : of all preictal windows, % model caught
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix,
)

# TensorFlow / Keras — our deep learning framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility seeds 
# Neural networks start with random weights
# Setting seeds makes the random initialization reproducible
# seed 42 = Run 1. Run 06_multi_run.py tries seeds 123,456,789,999
tf.random.set_seed(42)
np.random.seed(42)


# Configuration constants 
# All key settings in one place — easy to change and compare

# Path to the feature file created by 04_extract_features.py
FEATURES_PATH = Path("data/features/features_v6.npz")

# Where to save trained models and results
MODELS_DIR    = Path("models_v6")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Alert threshold: model must output >= 0.65 to fire an alert
# 0.65 = "65% confident this is preictal" before alerting
# Higher = fewer alerts, less false alarms, but misses more seizures
# Lower  = more alerts, catches more seizures, more false alarms
THRESHOLD  = 0.65

# Batch size: how many windows to process at once during training
# 256 is optimal for M4 Metal GPU — fills GPU memory efficiently
BATCH_SIZE = 256

# Maximum training epochs — early stopping usually kicks in before this
EPOCHS     = 100

# Step size used in preprocessing — needed to calculate FP/hour
# 5s step means there are 720 windows per hour of recording
STEP_SEC   = 5


# METRICS FUNCTION
# Calculates all performance metrics from predictions
def evaluate(y_true, y_prob, threshold=THRESHOLD):
    """
    Takes ground truth labels and model probabilities.
    Applies threshold to get binary predictions.
    Returns dict of all metrics.

    y_true : array of 0s and 1s (0=interictal, 1=preictal)
    y_prob : array of probabilities from model (0.0 to 1.0)
    """

    # Apply threshold to convert probabilities to binary predictions
    # e.g. 0.72 >= 0.65 → 1 (alert!), 0.45 < 0.65 → 0 (no alert)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix gives us TP, FP, FN, TN counts
    # TN = no alert, no seizure ✅
    # FP = alert fired, no seizure ❌ (false alarm)
    # FN = no alert, seizure happened ❌ (missed seizure)
    # TP = alert fired, seizure was coming ✅
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    # Calculate false alarms per hour of recording
    # Used to assess clinical usability (target: ≤2 per hour)
    n_inter      = int((y_true == 0).sum())
    record_hours = (n_inter * STEP_SEC) / 3600
    fp_per_hour  = fp / record_hours if record_hours > 0 else 0

    # FP per 10 TP — for every 10 seizures caught, how many false alarms?
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


# NEURAL NETWORK ARCHITECTURE
# Builds the model structure — does NOT train it yet
def build_nn(n_features):
    """
    Builds a 4-layer dense neural network.

    Architecture: 128 → 64 → 32 → 1
    This was proven best for cross-patient generalization
    on 14 patients with 64 features.

    BatchNormalization : normalizes layer outputs during training
                         makes training more stable and faster
    Dropout            : randomly disables neurons during training
                         prevents overfitting to training patients
    sigmoid output     : outputs probability between 0 and 1
    """

    # Input layer — expects 64 features per window
    inputs = keras.Input(shape=(n_features,))

    # Layer 1: 128 neurons
    # Dense(128) learns 128 different combinations of the 64 input features
    # BatchNorm stabilizes the distribution of outputs
    # Dropout(0.4) = randomly disable 40% of neurons each training step
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Layer 2: 64 neurons — learns combinations of layer 1 patterns
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Layer 3: 32 neurons — higher-level abstractions
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # Output layer: single neuron with sigmoid activation
    # Outputs probability 0.0-1.0 of being preictal
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=output)



# LOAD FEATURE DATA
print("\n" + "=" * 62)
print("  MODEL TRAINING v6  —  Run 1 (seed 42)")
print("=" * 62)
print(f"""
  Channels  : F7, T3, T5, C3, F8, T4, T6, C4
  Features  : 64 band powers
  Loss      : Binary crossentropy + class weights
  Arch      : 128->64->32->1
  Threshold : {THRESHOLD}
  Batch     : {BATCH_SIZE} (M4 Metal GPU)
  Epochs    : {EPOCHS} max, patience=12
""")

if not FEATURES_PATH.exists():
    print(f"Features not found: {FEATURES_PATH}")
    print("Run 04_extract_features.py first.")
    exit(1)

# Load the .npz file created by 04_extract_features.py
# X        : shape (n_windows, 64) — feature matrix
# y        : shape (n_windows,)    — labels (0=inter, 1=pre)
# patients : shape (n_windows,)    — patient ID per window
#            e.g. ["PN00","PN00",...,"PN03","PN03",...]
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



# LEAVE ONE PATIENT OUT (LOPO) TRAINING LOOP
# The core of our evaluation strategy
# LOPO means:
#   For each of the 14 patients in turn:
#     - Hold that patient out as the TEST set
#     - Train on all other 13 patients
#     - Evaluate on the held-out patient
#
# This is honest evaluation — the test patient's data is
# NEVER seen during training. It simulates a brand new
# patient putting on the device for the first time.
print("\n" + "=" * 62)
print("  LEAVE ONE PATIENT OUT")
print("=" * 62)

# Dictionary to store results for each test patient
nn_results = {}

for fold_idx, test_patient in enumerate(patient_ids):
    print(f"\n-- Fold {fold_idx+1}/{len(patient_ids)} "
          f"| Test: {test_patient} " + "-" * 25)

    # Create boolean masks to split data by patient
    # test_mask  = True for windows belonging to test_patient
    # train_mask = True for everyone else (13 patients)
    test_mask  = patients == test_patient
    train_mask = ~test_mask  # ~ means "NOT" (flip True/False)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    # Skip patients with no preictal data (PN01, PN11)
    # Can't evaluate a patient if we have nothing to detect
    if (y_test == 1).sum() == 0:
        print("  No preictal windows -- skipping")
        continue

    # spw = samples per window = imbalance ratio
    # e.g. 30:1 means 30 interictal windows per preictal window
    # Used as class_weight to compensate for imbalance
    n_pre   = int((y_train == 1).sum())
    n_inter = int((y_train == 0).sum())
    spw     = n_inter / n_pre

    print(f"  Train : {len(y_train):,} "
          f"(pre:{n_pre:,} | inter:{n_inter:,})  spw:{spw:.1f}")
    print(f"  Test  : {len(y_test):,} "
          f"(pre:{int((y_test==1).sum()):,} | "
          f"inter:{int((y_test==0).sum()):,})")

    # Feature scaling 
    # StandardScaler transforms features to mean=0, std=1
    # CRITICAL: fit ONLY on training data, then apply to test
    # Fitting on test data would be data leakage
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
    X_test_scaled  = scaler.transform(X_test)        # transform only

    print(f"  [NN] training (Metal GPU)...", end="", flush=True)

    # Build and compile the neural network 
    nn = build_nn(n_features)
    nn.compile(
        # Adam optimizer: adapts learning rate automatically
        # lr=1e-3 = 0.001 starting learning rate
        optimizer=keras.optimizers.Adam(1e-3),

        # Binary crossentropy: standard loss for yes/no problems
        # Stable and reliable — we tried focal loss but it
        # caused too much variance between runs
        loss="binary_crossentropy",

        # Track AUC during training to monitor progress
        # (used by EarlyStopping to decide when to stop)
        metrics=[keras.metrics.AUC(name="auc")],
    )

    # Train the model 
    nn.fit(
        X_train_scaled, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,

        # Hold out 15% of training data for validation
        # Validation data is used to monitor overfitting
        # and trigger early stopping — NOT used for training
        validation_split=0.15,

        # class_weight compensates for class imbalance
        # If 30:1 imbalance, preictal mistakes cost 30x more
        # Forces model to actually learn preictal patterns
        # instead of just predicting "normal" every time
        class_weight={0: 1.0, 1: spw},

        callbacks=[
            # EarlyStopping: stop training when val_auc stops
            # improving. patience=12 means wait 12 epochs
            # before giving up. restore_best_weights=True means
            # go back to the best weights we saw, not the last ones
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=12,
                restore_best_weights=True,
                verbose=0),

            # ReduceLROnPlateau: if loss stops improving,
            # reduce learning rate by half. Helps model find
            # better solutions when it gets stuck.
            # min_lr=1e-7 prevents LR from going too small
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-7, verbose=0),
        ],
        verbose=0,  # suppress per-epoch output (too much noise)
    )

    # Evaluate on held out test patient
    # This is the honest evaluation — patient never seen before
    prob    = nn.predict(X_test_scaled, verbose=0).flatten()
    metrics = evaluate(y_test, prob)
    nn_results[test_patient] = metrics

    grd = "✅" if metrics["auc_roc"] >= 0.70 else \
          "⚠️ " if metrics["auc_roc"] >= 0.60 else "❌"
    print(f"  {grd} AUC:{metrics['auc_roc']:.3f}  "
          f"Prec:{metrics['precision']:.3f}  "
          f"Rec:{metrics['recall']:.3f}  "
          f"TP:{metrics['tp']}  FP:{metrics['fp']}  "
          f"FP/hr:{metrics['fp_per_hour']:.1f}  "
          f"FP/10:{metrics['fp_per_10']}")

    # Save model and scaler 
    # Save the trained model weights for this patient
    # Used by 06_multi_run.py (to compare) and 07_finetune.py
    nn.save(str(MODELS_DIR / f"nn_{test_patient}.keras"))

    # Save the scaler fitted on this fold's training data
    # Must use the SAME scaler at inference time on device
    joblib.dump(scaler,
                str(MODELS_DIR / f"scaler_{test_patient}.pkl"))


# FINAL SUMMARY TABLE
aucs = [r["auc_roc"] for r in nn_results.values()]

print(f"\n{'=' * 68}")
print(f"  v6 RESULTS — Run 1 (seed 42)  threshold={THRESHOLD}")
print(f"{'=' * 68}")
print(f"  {'Patient':<10} {'AUC':>7} {'Prec':>7} {'Rec':>7} "
      f"{'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
      f"{'FP/10':>7}  Grade")
print("  " + "-" * 70)

for pat, r in sorted(nn_results.items()):
    auc = r["auc_roc"]
    grd = "✅ YES" if auc >= 0.70 else \
          "⚠️  MOD" if auc >= 0.60 else "❌ NO"
    print(f"  {pat:<10} {auc:>7.3f} {r['precision']:>7.3f} "
          f"{r['recall']:>7.3f} {r['f1']:>7.3f} "
          f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
          f"{r['fp_per_10']:>7}  {grd}")

print("  " + "-" * 70)
print(f"  {'MEAN':<10} {np.mean(aucs):>7.3f}")
print(f"{'=' * 68}")

pred  = sum(1 for a in aucs if a >= 0.70)
mod   = sum(1 for a in aucs if 0.60 <= a < 0.70)
poor  = sum(1 for a in aucs if a < 0.60)
total = len(aucs)
print(f"\n  ✅ Predictable : {pred}/{total}")
print(f"  ⚠️  Modest      : {mod}/{total}")
print(f"  ❌ Poor         : {poor}/{total}")
print(f"\n  Next: run 06_multi_run.py to try 4 more seeds")


# SAVE RESULTS
# Saved as JSON so 06_multi_run.py and 08_final_eval.py
with open(MODELS_DIR / "lopo_results.json", "w") as f:
    json.dump({"neural_net": nn_results,
               "config": {
                   "seed":      42,
                   "threshold": THRESHOLD,
                   "channels":  ["F7","T3","T5","C3",
                                  "F8","T4","T6","C4"],
                   "features":  64,
               }}, f, indent=2)
print(f"\n  Saved -> {MODELS_DIR}/lopo_results.json\n")