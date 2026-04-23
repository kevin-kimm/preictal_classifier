"""
=============================================================
  Siena Scalp EEG — Full Patient Review
  12_patient_review.py

  Generates a comprehensive review of model performance
  across all 14 patients comparing all pipeline versions.

  Shows:
    - Per patient AUC across all versions
    - Patient demographics vs predictability
    - Best model per patient
    - Overall ranking of patients by predictability
=============================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Load results from all versions 
RESULTS = {
    "v2_NN (64 feat)":  ("models/lopo_results.json",    "neural_net"),
    "v2_GB (148 feat)": ("models/lopo_results.json",    "gradient_boosting"),
    "v3_NN (287 feat)": ("models_v3/lopo_results.json", "neural_net"),
    "v3_GB (287 feat)": ("models_v3/lopo_results.json", "gradient_boosting"),
}

# Patient demographics 
DATA_ROOT    = Path("data/siena-scalp-eeg-database-1.0.0")
subject_info = pd.read_csv(DATA_ROOT / "subject_info.csv")
subject_info.columns = subject_info.columns.str.strip()

# Load all results 
all_results = {}
for label, (path, key) in RESULTS.items():
    p = Path(path)
    if p.exists():
        with open(str(p)) as f:
            data = json.load(f)
        if key in data:
            all_results[label] = data[key]
        else:
            print(f"⚠️  Key '{key}' not found in {path}")
    else:
        print(f"⚠️  File not found: {path}")

# Build comparison dataframe 
all_patients = sorted(subject_info["patient_id"].tolist())
records = []

for patient in all_patients:
    row = {"patient": patient}
    best_auc = 0
    best_model = "—"

    for label, results in all_results.items():
        if patient in results:
            auc = results[patient]["auc_roc"]
            row[label] = round(auc, 3)
            if auc > best_auc:
                best_auc  = auc
                best_model = label
        else:
            row[label] = None

    row["best_auc"]   = round(best_auc, 3) if best_auc > 0 else None
    row["best_model"] = best_model

    # Add demographics
    demo = subject_info[subject_info["patient_id"] == patient]
    if not demo.empty:
        row["age"]       = int(demo["age_years"].values[0])
        row["gender"]    = demo["gender"].values[0]
        row["sz_type"]   = demo["seizure"].values[0]
        row["n_seizures"]= int(demo["number_seizures"].values[0])
        row["rec_min"]   = int(demo["rec_time_minutes"].values[0])
    records.append(row)

df = pd.DataFrame(records)

# Print comprehensive table
print("\n" + "=" * 90)
print("  FULL PATIENT REVIEW — AUC ACROSS ALL PIPELINE VERSIONS")
print("=" * 90)

version_cols = list(all_results.keys())
print(f"  {'Patient':<8} {'Age':>4} {'Sex':>4} {'Type':>5} "
      f"{'Sz':>3} {'Rec':>5}", end="")
for v in version_cols:
    short = v[:12]
    print(f"  {short:>12}", end="")
print(f"  {'Best AUC':>9}  Best Model")
print("  " + "-" * 85)

for _, row in df.iterrows():
    predictable = ""
    if row.get("best_auc") and row["best_auc"] >= 0.70:
        predictable = " ✅"
    elif row.get("best_auc") and row["best_auc"] >= 0.60:
        predictable = " ⚠️"
    else:
        predictable = " ❌"

    print(f"  {row['patient']:<8} "
          f"{str(row.get('age','?')):>4} "
          f"{str(row.get('gender','?'))[:1]:>4} "
          f"{str(row.get('sz_type','?')):>5} "
          f"{str(row.get('n_seizures','?')):>3} "
          f"{str(row.get('rec_min','?')):>5}", end="")

    for v in version_cols:
        val = row.get(v)
        if val is None:
            print(f"  {'—':>12}", end="")
        else:
            print(f"  {val:>12.3f}", end="")

    best = row.get('best_auc')
    print(f"  {str(best) if best else '—':>9}{predictable}  "
          f"{row.get('best_model','—')}")

print("  " + "-" * 85)

# Means per version
print(f"  {'MEAN':<8} {'':>4} {'':>4} {'':>5} {'':>3} {'':>5}", end="")
for v in version_cols:
    vals = [row[v] for _, row in df.iterrows()
            if row.get(v) is not None]
    print(f"  {np.mean(vals):>12.3f}", end="")
print()

print(f"\n  ✅ AUC ≥ 0.70 = clinically meaningful prediction")
print(f"  ⚠️  AUC 0.60–0.70 = modest signal")
print(f"  ❌ AUC < 0.60 = poor / unpredictable")


# Predictability ranking 
print(f"\n{'=' * 90}")
print(f"  PATIENT PREDICTABILITY RANKING")
print(f"{'=' * 90}")

ranked = df[df["best_auc"].notna()].sort_values(
    "best_auc", ascending=False)

print(f"\n  {'Rank':<6} {'Patient':<10} {'Best AUC':>9} "
      f"{'Age':>4} {'Type':>6} {'Seizures':>9} "
      f"{'Recording':>10}  Best Model")
print("  " + "-" * 75)

for rank, (_, row) in enumerate(ranked.iterrows(), 1):
    grade = "✅ Predictable" if row["best_auc"] >= 0.70 else \
            "⚠️  Modest"      if row["best_auc"] >= 0.60 else \
            "❌ Unpredictable"
    print(f"  {rank:<6} {row['patient']:<10} "
          f"{row['best_auc']:>9.3f} "
          f"{str(row.get('age','?')):>4} "
          f"{str(row.get('sz_type','?')):>6} "
          f"{str(row.get('n_seizures','?')):>9} "
          f"{str(row.get('rec_min','?')):>8}m  "
          f"{grade}")

# Summary stats
predictable = (ranked["best_auc"] >= 0.70).sum()
modest      = ((ranked["best_auc"] >= 0.60) &
               (ranked["best_auc"] < 0.70)).sum()
poor        = (ranked["best_auc"] < 0.60).sum()

print(f"\n  Predictable (AUC ≥ 0.70) : {predictable} patients")
print(f"  Modest      (AUC 0.6–0.7): {modest} patients")
print(f"  Poor        (AUC < 0.60) : {poor} patients")
print(f"  No preictal data          : 2 patients (PN01, PN11)")



# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Patient Review — AUC Across All Pipeline Versions",
             fontsize=14, fontweight="bold")

# Left: grouped bar chart per patient
ax1 = axes[0]
patients_with_data = [p for p in all_patients
                      if any(df[df["patient"]==p][v].notna().any()
                             for v in version_cols)]

x      = np.arange(len(patients_with_data))
n_vers = len(version_cols)
width  = 0.18
colors = ["#2171B5", "#6BAED6", "#CB181D", "#FC8D59"]

for i, (v, color) in enumerate(zip(version_cols, colors)):
    vals = []
    for p in patients_with_data:
        row = df[df["patient"] == p]
        val = row[v].values[0] if not row.empty and row[v].notna().any() else 0
        vals.append(val if val else 0)
    ax1.bar(x + i * width, vals, width,
            label=v, color=color, alpha=0.85)

# Threshold lines
ax1.axhline(0.70, color="green", linewidth=1.5,
            linestyle="--", alpha=0.7, label="AUC 0.70 threshold")
ax1.axhline(0.50, color="gray", linewidth=1,
            linestyle=":", alpha=0.5, label="Random (0.50)")

ax1.set_xticks(x + width * (n_vers-1) / 2)
ax1.set_xticklabels(patients_with_data, rotation=45,
                    ha="right", fontsize=9)
ax1.set_ylabel("AUC-ROC", fontsize=10)
ax1.set_title("AUC per Patient per Pipeline Version",
              fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=8, loc="upper right")
ax1.spines[["top", "right"]].set_visible(False)

# Right: best AUC per patient with demographics 
ax2 = axes[1]
ranked_patients = ranked["patient"].tolist()
ranked_aucs     = ranked["best_auc"].tolist()
ranked_types    = ranked["sz_type"].tolist()

bar_colors = []
for auc in ranked_aucs:
    if auc >= 0.70:
        bar_colors.append("#2ca02c")
    elif auc >= 0.60:
        bar_colors.append("#ff7f0e")
    else:
        bar_colors.append("#d62728")

bars = ax2.barh(range(len(ranked_patients)),
                ranked_aucs, color=bar_colors, alpha=0.85)
ax2.axvline(0.70, color="green", linewidth=1.5,
            linestyle="--", alpha=0.7)
ax2.axvline(0.50, color="gray", linewidth=1,
            linestyle=":", alpha=0.5)

ax2.set_yticks(range(len(ranked_patients)))
ax2.set_yticklabels([f"{p} ({t})" for p, t in
                     zip(ranked_patients, ranked_types)],
                    fontsize=9)
ax2.set_xlabel("Best AUC (across all versions)", fontsize=10)
ax2.set_title("Patient Predictability Ranking\n(Best AUC across all pipeline versions)",
              fontweight="bold")
ax2.set_xlim(0, 1.0)
ax2.spines[["top", "right"]].set_visible(False)

for bar, auc in zip(bars, ranked_aucs):
    ax2.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
             f"{auc:.3f}", va="center", fontsize=8)

green_patch  = mpatches.Patch(color="#2ca02c", alpha=0.85,
                               label="Predictable (≥0.70)")
orange_patch = mpatches.Patch(color="#ff7f0e", alpha=0.85,
                               label="Modest (0.60–0.70)")
red_patch    = mpatches.Patch(color="#d62728", alpha=0.85,
                               label="Poor (<0.60)")
ax2.legend(handles=[green_patch, orange_patch, red_patch],
           fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("patient_review.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to patient_review.png")