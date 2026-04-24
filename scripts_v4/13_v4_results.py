"""
=============================================================
  Siena Scalp EEG — v4 Results Analysis
  13_v4_results.py

  Loads models_v4/lopo_results.json and generates:
    - Detailed per-patient metrics table
    - Success metrics summary
    - Comparison vs all previous versions
    - Saves results_v4.png
=============================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load results ───────────────────────────────────────────
V4_PATH = Path("models_v4/lopo_results.json")
V3_PATH = Path("models_v3/lopo_results.json")
V2_PATH = Path("models/lopo_results.json")

if not V4_PATH.exists():
    print("❌ models_v4/lopo_results.json not found")
    print("   Run scripts_v4/05_train_model.py first")
    exit(1)

with open(V4_PATH) as f: v4 = json.load(f)
v3 = json.load(open(V3_PATH)) if V3_PATH.exists() else {}
v2 = json.load(open(V2_PATH)) if V2_PATH.exists() else {}

gb_v4 = v4.get("gradient_boosting", {})
nn_v4 = v4.get("neural_net", {})
nn_v3 = v3.get("neural_net", {})
gb_v3 = v3.get("gradient_boosting", {})
nn_v2 = v2.get("neural_net", {})
gb_v2 = v2.get("gradient_boosting", {})

# Patient demographics
DATA_ROOT    = Path("data/siena-scalp-eeg-database-1.0.0")
subject_info = pd.read_csv(DATA_ROOT / "subject_info.csv")
subject_info.columns = subject_info.columns.str.strip()

all_patients = sorted(subject_info["patient_id"].tolist())


# ─────────────────────────────────────────────────────────────
# TABLE 1 — v4 DETAILED METRICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  v4 DETAILED RESULTS — 10-min preictal, 60-sec windows, 7-channel headband")
print("=" * 80)
print(f"\n  {'Patient':<8} {'Age':>4} {'Type':>6} "
      f"{'GB AUC':>8} {'NN AUC':>8} {'Best':>8} "
      f"{'Precision':>10} {'Recall':>8} {'F1':>6}  Grade")
print("  " + "-" * 75)

predictable = []
modest      = []
poor        = []

for patient in all_patients:
    demo = subject_info[subject_info["patient_id"] == patient]
    age  = int(demo["age_years"].values[0]) if not demo.empty else "?"
    sz_t = demo["seizure"].values[0] if not demo.empty else "?"

    gb_auc = gb_v4[patient]["auc_roc"] if patient in gb_v4 else None
    nn_auc = nn_v4[patient]["auc_roc"] if patient in nn_v4 else None

    if gb_auc is None and nn_auc is None:
        print(f"  {patient:<8} {str(age):>4} {sz_t:>6}  "
              f"{'—':>8} {'—':>8} {'—':>8}  No preictal data")
        continue

    best_auc   = max(v for v in [gb_auc, nn_auc] if v is not None)
    best_model = "GB" if gb_auc == best_auc else "NN"
    best_res   = (gb_v4 if best_model == "GB" else nn_v4)[patient]

    grade = "✅ Predictable" if best_auc >= 0.70 else \
            "⚠️  Modest"      if best_auc >= 0.60 else \
            "❌ Poor"

    if best_auc >= 0.70: predictable.append(patient)
    elif best_auc >= 0.60: modest.append(patient)
    else: poor.append(patient)

    print(f"  {patient:<8} {str(age):>4} {sz_t:>6} "
          f"  {gb_auc:>8.3f}" if gb_auc else f"  {'—':>8}",
          end="")
    print(f"  {nn_auc:>8.3f}" if nn_auc else f"  {'—':>8}",
          end="")
    print(f"  {best_auc:>8.3f} ({best_model})"
          f"  {best_res['precision']:>10.3f}"
          f"  {best_res['recall']:>8.3f}"
          f"  {best_res['f1']:>6.3f}  {grade}")

print("  " + "-" * 75)

# Means
gb_aucs = [gb_v4[p]["auc_roc"] for p in gb_v4]
nn_aucs = [nn_v4[p]["auc_roc"] for p in nn_v4]
print(f"\n  Mean GB AUC : {np.mean(gb_aucs):.3f}  (std: {np.std(gb_aucs):.3f})")
print(f"  Mean NN AUC : {np.mean(nn_aucs):.3f}  (std: {np.std(nn_aucs):.3f})")


# ─────────────────────────────────────────────────────────────
# TABLE 2 — SUCCESS METRICS SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  v4 SUCCESS METRICS SUMMARY")
print(f"{'=' * 80}")
print(f"\n  Predictable patients (AUC ≥ 0.70) : {len(predictable)}")
for p in predictable:
    best = max(gb_v4.get(p, {}).get("auc_roc", 0),
               nn_v4.get(p, {}).get("auc_roc", 0))
    print(f"    {p}  AUC {best:.3f}")

print(f"\n  Modest patients (AUC 0.60–0.70)   : {len(modest)}")
for p in modest:
    best = max(gb_v4.get(p, {}).get("auc_roc", 0),
               nn_v4.get(p, {}).get("auc_roc", 0))
    print(f"    {p}  AUC {best:.3f}")

print(f"\n  Poor patients (AUC < 0.60)         : {len(poor)}")
for p in poor:
    best = max(gb_v4.get(p, {}).get("auc_roc", 0),
               nn_v4.get(p, {}).get("auc_roc", 0))
    print(f"    {p}  AUC {best:.3f}")

print(f"\n  No preictal data                   : PN01, PN11")
print(f"\n  Detection rate : {len(predictable)}/{len(predictable)+len(modest)+len(poor)} "
      f"patients predictable ({len(predictable)/(len(predictable)+len(modest)+len(poor))*100:.0f}%)")


# ─────────────────────────────────────────────────────────────
# TABLE 3 — CROSS VERSION COMPARISON
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  CROSS VERSION COMPARISON — Best NN AUC per patient")
print(f"{'=' * 80}")
print(f"\n  {'Patient':<10} {'v2 NN':>8} {'v3 NN':>8} {'v4 NN':>8} "
      f"{'Best':>8}  Change v3→v4")
print("  " + "-" * 60)

for patient in all_patients:
    v2_auc = nn_v2.get(patient, {}).get("auc_roc")
    v3_auc = nn_v3.get(patient, {}).get("auc_roc")
    v4_auc = nn_v4.get(patient, {}).get("auc_roc")

    if v2_auc is None and v3_auc is None and v4_auc is None:
        print(f"  {patient:<10} {'—':>8} {'—':>8} {'—':>8}  No preictal data")
        continue

    vals    = [v for v in [v2_auc, v3_auc, v4_auc] if v is not None]
    best    = max(vals)
    change  = ""
    if v3_auc and v4_auc:
        diff   = v4_auc - v3_auc
        arrow  = "↑" if diff > 0.02 else "↓" if diff < -0.02 else "→"
        change = f"{arrow} {abs(diff):.3f}"

    print(f"  {patient:<10} "
          f"  {v2_auc:.3f}" if v2_auc else f"  {'—':>8}",
          end="")
    print(f"  {v3_auc:.3f}" if v3_auc else f"  {'—':>8}",
          end="")
    print(f"  {v4_auc:.3f}" if v4_auc else f"  {'—':>8}",
          end="")
    print(f"  {best:.3f}  {change}")

print("  " + "-" * 60)
v2m = np.mean([nn_v2[p]["auc_roc"] for p in nn_v2])
v3m = np.mean([nn_v3[p]["auc_roc"] for p in nn_v3]) if nn_v3 else 0
v4m = np.mean([nn_v4[p]["auc_roc"] for p in nn_v4])
print(f"  {'MEAN':<10} {v2m:>8.3f} {v3m:>8.3f} {v4m:>8.3f}")


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("v4 Results — 10-min Preictal, 60-sec Windows, 7-channel Headband",
             fontsize=13, fontweight="bold")

patients_with_data = [p for p in all_patients
                      if p in gb_v4 or p in nn_v4]

# ── Left: GB vs NN AUC comparison ─────────────────────────
ax1 = axes[0]
x      = np.arange(len(patients_with_data))
width  = 0.35

gb_vals = [gb_v4.get(p, {}).get("auc_roc", 0) for p in patients_with_data]
nn_vals = [nn_v4.get(p, {}).get("auc_roc", 0) for p in patients_with_data]

bars1 = ax1.bar(x - width/2, gb_vals, width,
                label="GradientBoosting", color="#2171B5", alpha=0.85)
bars2 = ax1.bar(x + width/2, nn_vals, width,
                label="Neural Network", color="#CB181D", alpha=0.85)

ax1.axhline(0.70, color="green", linewidth=1.5,
            linestyle="--", alpha=0.7, label="AUC 0.70 threshold")
ax1.axhline(0.50, color="gray", linewidth=1,
            linestyle=":", alpha=0.5, label="Random (0.50)")

ax1.set_xticks(x)
ax1.set_xticklabels(patients_with_data, rotation=45,
                    ha="right", fontsize=9)
ax1.set_ylabel("AUC-ROC", fontsize=10)
ax1.set_title("v4: GradientBoosting vs Neural Network per Patient",
              fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

# ── Right: v2 vs v3 vs v4 NN comparison ───────────────────
ax2 = axes[1]
x2     = np.arange(len(patients_with_data))
width2 = 0.25

v2_vals = [nn_v2.get(p, {}).get("auc_roc", 0) for p in patients_with_data]
v3_vals = [nn_v3.get(p, {}).get("auc_roc", 0) for p in patients_with_data] if nn_v3 else [0]*len(patients_with_data)
v4_vals = [nn_v4.get(p, {}).get("auc_roc", 0) for p in patients_with_data]

ax2.bar(x2 - width2, v2_vals, width2,
        label="v2 NN (8ch, 5min, 30s)", color="#4292C6", alpha=0.85)
ax2.bar(x2,          v3_vals, width2,
        label="v3 NN (7ch, 5min, 30s)", color="#F16913", alpha=0.85)
ax2.bar(x2 + width2, v4_vals, width2,
        label="v4 NN (7ch, 10min, 60s)", color="#CB181D", alpha=0.85)

ax2.axhline(0.70, color="green", linewidth=1.5,
            linestyle="--", alpha=0.7, label="AUC 0.70 threshold")
ax2.axhline(0.50, color="gray", linewidth=1,
            linestyle=":", alpha=0.5)

ax2.set_xticks(x2)
ax2.set_xticklabels(patients_with_data, rotation=45,
                    ha="right", fontsize=9)
ax2.set_ylabel("AUC-ROC", fontsize=10)
ax2.set_title("Neural Network AUC: v2 vs v3 vs v4",
              fontweight="bold")
ax2.set_ylim(0, 1.0)
ax2.legend(fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results_v4.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Saved to results_v4.png")