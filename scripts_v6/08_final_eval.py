"""
=============================================================
  Siena Scalp EEG — Final Evaluation Dashboard v6
  08_final_eval.py

  Loads all result files and prints a complete comparison:
    Run 1     : seed 42 baseline
    Multi-run : best of 5 seeds per patient
    Fine-tuned: patient-specific final layers
=============================================================
"""


import json
import numpy as np
from pathlib import Path

# All result files live in models_v6/
MODELS_DIR = Path("models_v6")


# Helper to load JSON files 
def load(path, key=None):
    """
    Loads a JSON result file.
    If key is provided, returns just that section.
    If file doesn't exist, returns empty dict (graceful failure).
    """
    p = Path(path)
    if not p.exists():
        return {}  # file not found — return empty rather than crash
    with open(p) as f:
        d = json.load(f)
    return d.get(key, d) if key else d


# Load all results from all three training stages 

# Run 1 results: saved by 05_train_model.py
# Contains AUC, TP, FP, etc. for each patient with seed 42
run1 = load(MODELS_DIR / "lopo_results.json", "neural_net")

# Multi-run results: saved by 06_multi_run.py
# Contains the BEST AUC per patient across all 5 seeds
multi = load(MODELS_DIR / "multi_run_results.json", "best_aucs")

# Fine-tuning results: saved by 07_finetune.py
# Contains both base (before) and finetuned (after) AUCs
ft_base  = load(MODELS_DIR / "finetuned_results.json", "base")
ft_final = load(MODELS_DIR / "finetuned_results.json", "finetuned")

# Get sorted list of all patients across all result files
all_patients = sorted(set(list(run1.keys()) + list(multi.keys())))


# COMPARISON TABLE
# Shows all 3 stages side by side per patient
# This is the complete story of v6:
#   Run 1    = what one training run achieves off the shelf
#   Multi    = what the best seed achieves
#   Finetune = what patient-specific calibration achieves
print("\n" + "=" * 76)
print("  v6 FINAL RESULTS DASHBOARD")
print("=" * 76)
print(f"""
  Col 1 — Run 1      : seed 42, single training run
  Col 2 — Multi-run  : best AUC across 5 seeds (42,123,456,789,999)
  Col 3 — Fine-tuned : final 2 layers adapted to patient's own data
                       (simulates calibration after device deployment)

  Grading: ✅ AUC >= 0.70 (predictable)
           ⚠️  AUC 0.60-0.70 (modest)
           ❌ AUC < 0.60 (poor)
""")

print(f"  {'Patient':<10} {'Run 1':>8} {'Multi':>8} "
      f"{'Fine-tune':>10}  Best  Grade")
print("  " + "-" * 56)

# Lists for computing means at the bottom
all_run1  = []
all_multi = []
all_ft    = []

for pat in all_patients:
    # Get AUC from each stage (None if not available)
    c1 = run1.get(pat,    {}).get("auc_roc")   # Run 1
    c2 = multi.get(pat)                         # Multi-run best
    c3 = ft_final.get(pat, {}).get("auc_roc")  # Fine-tuned

    # Find the best AUC across all stages
    vals = [v for v in [c1, c2, c3] if v is not None]
    if not vals: continue
    best = max(vals)
    grd  = "✅" if best >= 0.70 else "⚠️ " if best >= 0.60 else "❌"

    # Format: show "—" if a stage wasn't run yet
    def fmt(v):
        return f"{v:>8.3f}" if v is not None else f"{'—':>8}"

    print(f"  {pat:<10} {fmt(c1)} {fmt(c2)} {fmt(c3)}  "
          f"{best:.3f}  {grd}")

    # Accumulate for means
    if c1 is not None: all_run1.append(c1)
    if c2 is not None: all_multi.append(c2)
    if c3 is not None: all_ft.append(c3)

print("  " + "-" * 56)

# Print mean and predictable count for each stage
configs = [
    ("Run 1",      all_run1),
    ("Multi-run",  all_multi),
    ("Fine-tuned", all_ft),
]
for label, aucs in configs:
    if aucs:
        pred = sum(1 for a in aucs if a >= 0.70)
        print(f"  {label:<12} mean={np.mean(aucs):.3f}  "
              f"predictable={pred}/{len(aucs)}")

print(f"\n{'=' * 76}\n")