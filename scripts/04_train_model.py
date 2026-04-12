"""
=============================================================
  Siena Scalp EEG — CNN+LSTM Model Training
  04_train_model.py

  Architecture : CNN + LSTM hybrid
  Validation   : Leave-One-Patient-Out (LOPO)
  Imbalance    : Handled via class weights
  Metrics      : F1, AUC-ROC, Precision, Recall (NOT accuracy)

  For each fold:
    - Train on 13 patients
    - Test on 1 held-out patient
    - Save best model + results

  Output folder: models/
=============================================================
"""

import numpy as np
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# Config
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_CHANNELS    = 6
N_SAMPLES     = 7500    # 30s × 250 Hz
BATCH_SIZE    = 32
EPOCHS        = 30
THRESHOLD     = 0.65    # decision threshold higher = more precise

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)



# Data load
def load_patient(npz_path: Path) -> tuple:
    """Load X, y for one patient from .npz file."""
    data = np.load(str(npz_path))
    X = data["X"]  # (N, 6, 7500)
    y = data["y"]  # (N,)
    return X.astype(np.float32), y.astype(np.int8)


def load_all_patients(processed_dir: Path) -> dict:
    """Load all patient data into a dictionary keyed by patient ID."""
    patients = {}
    for npz_path in sorted(processed_dir.glob("*.npz")):
        patient_id = npz_path.stem
        X, y = load_patient(npz_path)
        patients[patient_id] = (X, y)
        n_pre   = int((y == 1).sum())
        n_inter = int((y == 0).sum())
        print(f"  Loaded {patient_id}: {len(y):>5} windows "
              f"| preictal: {n_pre:>4} | interictal: {n_inter:>5}")
    return patients


# MODEL ARCHITECTURE — CNN + LSTM hybrid
#
# Input shape: (6, 7500)  →  6 channels, 7500 time points
#
# CNN block: extracts local temporal features per channel
#   - Conv1D scans across time, learning patterns like
#     high-frequency bursts, rhythmic activity, spikes
#   - MaxPooling reduces sequence length so LSTM is manageable
#
# LSTM block: learns how features evolve over 30 seconds
#   - Captures longer temporal dependencies
#   - Bidirectional = looks forward AND backward in window
#
# Output: single sigmoid neuron = P(preictal)
def build_model(n_channels: int, n_samples: int) -> keras.Model:
    inputs = keras.Input(shape=(n_channels, n_samples),
                         name="eeg_input")

    # Transpose to (time_steps, channels) for Conv1D 
    x = layers.Permute((2, 1))(inputs)  # → (7500, 6)

    # CNN Block 1 
    x = layers.Conv1D(filters=32, kernel_size=5,
                      activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)   # → (1875, 32)
    x = layers.Dropout(0.3)(x)

    # CNN Block 2 
    x = layers.Conv1D(filters=64, kernel_size=5,
                      activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=5)(x)   # → (375, 64)
    x = layers.Dropout(0.3)(x)

    # CNN Block 3 
    x = layers.Conv1D(filters=128, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=5)(x)   # → (75, 128)
    x = layers.Dropout(0.3)(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True)
        )(x)                                  # → (75, 128)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False)
        )(x)                                  # → (64,)
    x = layers.Dropout(0.3)(x)

    # Classification head
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid",
                          name="preictal_prob")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


# Metrics helper
def evaluate(y_true, y_prob, threshold=THRESHOLD) -> dict:
    """Compute all metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    return {
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "n_preictal":   int((y_true == 1).sum()),
        "n_interictal": int((y_true == 0).sum()),
    }


# Leave one patient out training 
def run_lopo(patients: dict):
    """
    For each patient:
      - Use that patient as the test set
      - Train on all other patients
      - Evaluate and save results
    """
    patient_ids = sorted(patients.keys())
    all_results = {}

    print("\n" + "=" * 62)
    print("  LEAVE-ONE-PATIENT-OUT CROSS VALIDATION")
    print("=" * 62)

    for fold_idx, test_patient in enumerate(patient_ids):
        print(f"\n── Fold {fold_idx+1}/{len(patient_ids)} "
              f"| Test patient: {test_patient} " + "─" * 20)

        # Split data 
        train_patients = [p for p in patient_ids if p != test_patient]

        X_train_list, y_train_list = [], []
        for p in train_patients:
            X_p, y_p = patients[p]
            X_train_list.append(X_p)
            y_train_list.append(y_p)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_test, y_test = patients[test_patient]

        print(f"  Train: {len(y_train):,} windows "
              f"(preictal: {int((y_train==1).sum()):,} | "
              f"interictal: {int((y_train==0).sum()):,})")
        print(f"  Test : {len(y_test):,} windows "
              f"(preictal: {int((y_test==1).sum()):,} | "
              f"interictal: {int((y_test==0).sum()):,})")

        # Skip if test patient has no preictal windows
        if (y_test == 1).sum() == 0:
            print(f"{test_patient} has no preictal windows — skipping fold")
            continue

        # Compute class weights 
        classes      = np.array([0, 1])
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weight_dict = {0: class_weights_arr[0],
                             1: class_weights_arr[1]}
        print(f"  Class weights: "
              f"interictal={class_weight_dict[0]:.2f}, "
              f"preictal={class_weight_dict[1]:.2f}")

        # Build fresh model for each fold
        model = build_model(N_CHANNELS, N_SAMPLES)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     keras.metrics.AUC(name="auc"),
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall")]
        )

        # Callbacks 
        model_path = MODELS_DIR / f"model_{test_patient}.keras"
        callbacks = [
            # Save best model based on val AUC
            keras.callbacks.ModelCheckpoint(
                str(model_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=0
            ),
            # Stop early if no improvement for 5 epochs
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce LR on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            ),
        ]

        # Train 
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.15,    # 15% of train for validation
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on held-out test patient 
        y_prob = model.predict(X_test, verbose=0).flatten()
        results = evaluate(y_test, y_prob, threshold=THRESHOLD)
        results["test_patient"] = test_patient
        results["epochs_trained"] = len(history.history["loss"])

        all_results[test_patient] = results

        print(f"\n  ── Results for {test_patient} ──────────────────")
        print(f"  AUC-ROC   : {results['auc_roc']:.3f}")
        print(f"  F1        : {results['f1']:.3f}")
        print(f"  Precision : {results['precision']:.3f}  "
              f"← how often alert is correct")
        print(f"  Recall    : {results['recall']:.3f}  "
              f"← how many seizures caught")
        print(f"  Confusion matrix:")
        cm = results["confusion_matrix"]
        print(f"    TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
        print(f"    FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")

        # Save per-fold results
        with open(MODELS_DIR / f"results_{test_patient}.json", "w") as f:
            json.dump(results, f, indent=2)

    return all_results


# Aggregate results across all folds 
def summarize_results(all_results: dict):
    if not all_results:
        print("No results to summarize.")
        return

    aucs       = [r["auc_roc"]   for r in all_results.values()]
    f1s        = [r["f1"]        for r in all_results.values()]
    precisions = [r["precision"] for r in all_results.values()]
    recalls    = [r["recall"]    for r in all_results.values()]

    print("\n" + "=" * 62)
    print("  LOPO SUMMARY — ALL FOLDS")
    print("=" * 62)
    print(f"  {'Patient':<10} {'AUC':>7} {'F1':>7} "
          f"{'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 48)
    for patient, r in sorted(all_results.items()):
        print(f"  {patient:<10} "
              f"{r['auc_roc']:>7.3f} "
              f"{r['f1']:>7.3f} "
              f"{r['precision']:>10.3f} "
              f"{r['recall']:>8.3f}")

    print("  " + "-" * 48)
    print(f"  {'MEAN':<10} "
          f"{np.mean(aucs):>7.3f} "
          f"{np.mean(f1s):>7.3f} "
          f"{np.mean(precisions):>10.3f} "
          f"{np.mean(recalls):>8.3f}")
    print(f"  {'STD':<10} "
          f"{np.std(aucs):>7.3f} "
          f"{np.std(f1s):>7.3f} "
          f"{np.std(precisions):>10.3f} "
          f"{np.std(recalls):>8.3f}")
    print("=" * 62)

    # Save full summary
    summary = {
        "mean_auc":       float(np.mean(aucs)),
        "mean_f1":        float(np.mean(f1s)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall":    float(np.mean(recalls)),
        "std_auc":        float(np.std(aucs)),
        "per_patient":    all_results,
    }
    with open(MODELS_DIR / "lopo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {MODELS_DIR}/lopo_summary.json")


# Main
if __name__ == "__main__":

    # Print model summary once before training
    print("\nModel Architecture:")
    demo_model = build_model(N_CHANNELS, N_SAMPLES)
    demo_model.summary()
    total_params = demo_model.count_params()
    print(f"Total parameters: {total_params:,}\n")

    # Load all patient data
    print("\nLoading preprocessed data...")
    patients = load_all_patients(PROCESSED_DIR)
    print(f"\nLoaded {len(patients)} patients\n")

    # Run LOPO training
    all_results = run_lopo(patients)

    # Print final summary
    summarize_results(all_results)