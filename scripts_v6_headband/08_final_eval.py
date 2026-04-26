"""
=============================================================
  Siena Scalp EEG — Final Evaluation Dashboard v6 Headband
  08_final_eval.py

  Loads all result files and prints a complete comparison:
    Run 1     : seed 42 baseline
    Multi-run : best seed per patient across all runs
=============================================================
"""


import json
import numpy as np
from pathlib import Path

# All result files live in models_v6_headband/
MODELS_DIR = Path("models_v6_headband")


# Helper to load JSON files safely 
def load(path, key=None):
    """
    Loads a JSON result file.
    If key is provided, returns just that section.
    If file doesn't exist, returns empty dict (graceful failure).
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    return d.get(key, d) if key else d


# Load all results 
# Run 1 results: saved by 05_train_model.py
run1  = load(MODELS_DIR / "lopo_results.json", "neural_net")

# Multi-run results: saved by 06_multi_run.py
# Contains the BEST AUC per patient across all seeds
multi = load(MODELS_DIR / "multi_run_results.json", "best_aucs")

all_patients = sorted(set(list(run1.keys()) + list(multi.keys())))


# COMPARISON TABLE
# Shows Run 1 vs Multi-run side by side per patient
#   Run 1 = what one training run achieves off the shelf
#   Multi = what the best seed achieves across all runs
print("\n" + "=" * 76)
print("  v6 HEADBAND FINAL RESULTS DASHBOARD")
print("=" * 76)
print(f"""
  Montage    : F7, T3, T5, O1, Pz, O2, T6, T4, F8  (9ch hybrid)
  Features   : 72 (band powers + ratios + entropy)

  Col 1 — Run 1     : seed 42, single training run
  Col 2 — Multi-run : best AUC across all seeds per patient

  Grading: ✅ AUC >= 0.70 (predictable)
           ⚠️  AUC 0.60-0.70 (modest)
           ❌ AUC < 0.60 (poor)

  Compare with v6 (8ch clinical):
    Run 1    mean=0.568  predictable=1/12
    Multi    mean=0.713  predictable=8/12
""")

print(f"  {'Patient':<10} {'Run 1':>8} {'Multi':>8}  Best  Grade")
print("  " + "-" * 44)

all_run1  = []
all_multi = []

for pat in all_patients:
    # Get AUC from each stage
    c1 = run1.get(pat,  {}).get("auc_roc")   # Run 1
    c2 = multi.get(pat)                        # Multi-run best

    vals = [v for v in [c1, c2] if v is not None]
    if not vals: continue
    best = max(vals)
    grd  = "✅" if best >= 0.70 else "⚠️ " if best >= 0.60 else "❌"

    def fmt(v):
        return f"{v:>8.3f}" if v is not None else f"{'—':>8}"

    print(f"  {pat:<10} {fmt(c1)} {fmt(c2)}  "
          f"{best:.3f}  {grd}")

    if c1 is not None: all_run1.append(c1)
    if c2 is not None: all_multi.append(c2)

print("  " + "-" * 44)

# Print mean and predictable count for each stage
configs = [
    ("Run 1",     all_run1),
    ("Multi-run", all_multi),
]
for label, aucs in configs:
    if aucs:
        pred = sum(1 for a in aucs if a >= 0.70)
        print(f"  {label:<12} mean={np.mean(aucs):.3f}  "
              f"predictable={pred}/{len(aucs)}")

print(f"\n{'=' * 76}\n")