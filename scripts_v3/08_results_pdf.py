"""
=============================================================
  Siena Scalp EEG — Results PDF Report
  08_results_pdf.py

  Generates a single PDF containing:
    - Pipeline overview and metric glossary
    - Per-patient results tables (GB + NN)
    - All confusion matrix images
    - Summary statistics

  Output: models_v3/v3_results_report.pdf
=============================================================
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)

MODELS_DIR = Path("models_v3")
CM_DIR     = MODELS_DIR / "confusion_matrices"
OUTPUT_PDF = MODELS_DIR / "v3_results_report.pdf"

# ── Load results ──────────────────────────────────────────
eval_path = MODELS_DIR / "eval_results.json"
if not eval_path.exists():
    print("Run 07_evaluate.py first.")
    exit(1)

with open(eval_path) as f:
    data = json.load(f)

gb_results = data.get("gradient_boosting", {})
nn_results = data.get("neural_net", {})

def fp_per_10(tp, fp):
    if tp == 0: return "inf"
    return f"{fp/tp*10:.1f}"

def grade(auc):
    if auc >= 0.70: return "Predictable"
    if auc >= 0.60: return "Modest"
    return "Poor"

# ── Styles ────────────────────────────────────────────────
styles  = getSampleStyleSheet()
title_s = ParagraphStyle("title_s", parent=styles["Title"],
                          fontSize=20, spaceAfter=6)
h1_s    = ParagraphStyle("h1_s", parent=styles["Heading1"],
                          fontSize=14, spaceBefore=14, spaceAfter=6,
                          textColor=colors.HexColor("#1a3a5c"))
h2_s    = ParagraphStyle("h2_s", parent=styles["Heading2"],
                          fontSize=11, spaceBefore=10, spaceAfter=4,
                          textColor=colors.HexColor("#2171B5"))
body_s  = ParagraphStyle("body_s", parent=styles["Normal"],
                          fontSize=9, spaceAfter=4, leading=13)
mono_s  = ParagraphStyle("mono_s", parent=styles["Code"],
                          fontSize=8, spaceAfter=2, leading=11)
small_s = ParagraphStyle("small_s", parent=styles["Normal"],
                          fontSize=8, spaceAfter=2)

def tbl_style(header_color="#1a3a5c"):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor(header_color)),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 8),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,0), (0,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ])

# ── Build PDF ─────────────────────────────────────────────
story = []
doc   = SimpleDocTemplate(
    str(OUTPUT_PDF), pagesize=letter,
    leftMargin=0.75*inch, rightMargin=0.75*inch,
    topMargin=0.75*inch, bottomMargin=0.75*inch
)

# ── Cover ─────────────────────────────────────────────────
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("Seizure Prediction Algorithm", title_s))
story.append(Paragraph("v3 Pipeline — Evaluation Results", h1_s))
story.append(HRFlowable(width="100%", thickness=2,
                         color=colors.HexColor("#1a3a5c")))
story.append(Spacer(1, 0.2*inch))

cover_data = [
    ["Parameter", "Value"],
    ["Pipeline",  "scripts_v3"],
    ["Channels",  "7  —  T3, T5, O1, Pz, O2, T6, T4  (headband)"],
    ["Window",    "30 seconds per prediction"],
    ["Preictal",  "5 minutes before seizure onset"],
    ["Features",  "287  (band powers, PLV, coherence, Hjorth, entropy)"],
    ["Models",    "GradientBoosting + Neural Network (focal loss a=0.75)"],
    ["Threshold", "0.65  (model must be >= 65% confident to fire alert)"],
    ["Validation","Leave-One-Patient-Out (LOPO) — 12 folds"],
    ["Patients",  "14 total  |  12 evaluable  |  PN01, PN11 no preictal data"],
]
t = Table(cover_data, colWidths=[1.8*inch, 5.2*inch])
t.setStyle(tbl_style())
story.append(t)
story.append(PageBreak())


# ── Metric Glossary ───────────────────────────────────────
story.append(Paragraph("Metric Glossary", h1_s))
story.append(HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor("#2171B5")))
story.append(Spacer(1, 0.1*inch))

glossary = [
    ["Metric", "Range", "What it means", "Target"],
    ["AUC-ROC",
     "0.5 – 1.0",
     "How well model ranks preictal above interictal across all "
     "thresholds. 0.5 = random, 1.0 = perfect.",
     ">= 0.70"],
    ["AUC-PR",
     "0.0 – 1.0",
     "Like AUC-ROC but focused on the rare preictal class. "
     "Random baseline ~0.03.",
     "> 0.10"],
    ["Precision",
     "0.0 – 1.0",
     "Of all alerts fired, what fraction were correct. "
     "0.66 = 34% are false alarms.",
     "> 0.50"],
    ["Recall",
     "0.0 – 1.0",
     "Of all preictal windows, what fraction did the model catch. "
     "0.80 = missed 20%.",
     "> 0.50"],
    ["F1",
     "0.0 – 1.0",
     "Harmonic mean of precision and recall. Balances both.",
     "> 0.30"],
    ["TP", "count",
     "True Positive — alert fired, seizure was coming.", "Higher"],
    ["FP", "count",
     "False Positive — alert fired, no seizure (false alarm).", "Lower"],
    ["FN", "count",
     "False Negative — no alert, seizure happened (missed).", "Lower"],
    ["FP/10", "ratio",
     "False alarms per 10 seizures correctly caught. "
     "inf = caught 0 seizures.", "<= 5"],
]
gt = Table(glossary, colWidths=[0.9*inch, 0.7*inch, 4.0*inch, 0.8*inch])
gt.setStyle(tbl_style())
story.append(gt)
story.append(PageBreak())


# ── Results table helper ──────────────────────────────────
def results_table(results, model_name, color):
    story.append(Paragraph(f"{model_name} — Test Results", h2_s))
    header = ["Patient", "AUC-ROC", "AUC-PR", "Recall",
              "Precision", "F1", "TP", "FP", "FN", "FP/10", "Grade"]
    rows   = [header]

    aucs = []
    for pat, r in sorted(results.items()):
        auc  = r["auc_roc"]
        aucs.append(auc)
        grd  = grade(auc)
        rows.append([
            pat,
            f"{auc:.3f}",
            f"{r['auc_pr']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['f1']:.3f}",
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
            fp_per_10(r["tp"], r["fp"]),
            grd,
        ])

    # Mean row
    rows.append([
        "MEAN",
        f"{np.mean(aucs):.3f}", "—", "—", "—", "—",
        "—", "—", "—", "—", "—"
    ])

    colw = [0.65*inch, 0.65*inch, 0.6*inch, 0.6*inch,
            0.75*inch, 0.55*inch, 0.4*inch, 0.4*inch,
            0.4*inch, 0.5*inch, 0.85*inch]
    t = Table(rows, colWidths=colw)
    ts = tbl_style(color)

    # Color grade cells
    for i, row in enumerate(rows[1:], start=1):
        if len(row) > 10:
            grd = row[10]
            if grd == "Predictable":
                ts.add("BACKGROUND", (10, i), (10, i),
                        colors.HexColor("#c7e9c0"))
                ts.add("TEXTCOLOR", (10, i), (10, i),
                        colors.HexColor("#1a6b1a"))
            elif grd == "Modest":
                ts.add("BACKGROUND", (10, i), (10, i),
                        colors.HexColor("#fdd9a0"))
            elif grd == "Poor":
                ts.add("BACKGROUND", (10, i), (10, i),
                        colors.HexColor("#fcbba1"))

    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.1*inch))

    # Grade summary
    pred = sum(1 for r in results.values() if r["auc_roc"] >= 0.70)
    mod  = sum(1 for r in results.values()
               if 0.60 <= r["auc_roc"] < 0.70)
    poor = sum(1 for r in results.values() if r["auc_roc"] < 0.60)
    n    = len(results)
    story.append(Paragraph(
        f"<b>Predictable (AUC >= 0.70):</b> {pred}/{n} &nbsp;&nbsp; "
        f"<b>Modest (0.60-0.70):</b> {mod}/{n} &nbsp;&nbsp; "
        f"<b>Poor (&lt;0.60):</b> {poor}/{n}",
        body_s))
    story.append(Spacer(1, 0.15*inch))


# ── Results pages ─────────────────────────────────────────
story.append(Paragraph("Per-Patient Test Results", h1_s))
story.append(HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor("#2171B5")))
story.append(Spacer(1, 0.1*inch))

results_table(gb_results, "Gradient Boosting", "#1a3a5c")
results_table(nn_results, "Neural Network",    "#6b1a1a")


# ── Head to head ──────────────────────────────────────────
story.append(Paragraph("Head-to-Head Comparison", h2_s))
common = sorted(set(gb_results) & set(nn_results))
h2h_data = [["Patient", "GB AUC", "NN AUC", "Best AUC", "Winner", "Grade"]]
for pat in common:
    ga = gb_results[pat]["auc_roc"]
    na = nn_results[pat]["auc_roc"]
    if ga > na:
        winner, best = "GradBoost", ga
    else:
        winner, best = "Neural net", na
    h2h_data.append([pat, f"{ga:.3f}", f"{na:.3f}",
                      f"{best:.3f}", winner, grade(best)])

ht = Table(h2h_data,
           colWidths=[0.8*inch, 0.9*inch, 0.9*inch,
                      0.9*inch, 1.1*inch, 1.1*inch])
ht.setStyle(tbl_style())
story.append(ht)
story.append(PageBreak())


# ── AUC comparison chart ──────────────────────────────────
story.append(Paragraph("AUC-ROC Comparison Chart", h1_s))
story.append(HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor("#2171B5")))
story.append(Spacer(1, 0.1*inch))

patients = sorted(set(gb_results) & set(nn_results))
gb_aucs  = [gb_results[p]["auc_roc"] for p in patients]
nn_aucs  = [nn_results[p]["auc_roc"] for p in patients]

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(patients))
w = 0.35
ax.bar(x - w/2, gb_aucs, w, label="GradientBoosting",
       color="#2171B5", alpha=0.85)
ax.bar(x + w/2, nn_aucs, w, label="Neural Network",
       color="#CB181D", alpha=0.85)
ax.axhline(0.70, color="green", linewidth=1.5,
           linestyle="--", label="Target 0.70")
ax.axhline(0.50, color="gray", linewidth=1,
           linestyle=":", alpha=0.6, label="Random 0.50")
ax.set_xticks(x)
ax.set_xticklabels(patients, fontsize=9)
ax.set_ylabel("AUC-ROC", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("AUC-ROC per Patient — v3 Pipeline", fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close()
buf.seek(0)
story.append(Image(buf, width=6.5*inch, height=3*inch))
story.append(PageBreak())


# ── Confusion matrices ────────────────────────────────────
story.append(Paragraph("Confusion Matrices — All Patients", h1_s))
story.append(HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor("#2171B5")))
story.append(Paragraph(
    "Each matrix shows model predictions vs actual labels "
    "for the held-out test patient. "
    "TN=top-left, FP=top-right, FN=bottom-left, TP=bottom-right.",
    body_s))
story.append(Spacer(1, 0.1*inch))

# 2 confusion matrices per row (GB + NN per patient)
img_w = 2.8*inch
img_h = 2.5*inch

for pat in sorted(set(gb_results) & set(nn_results)):
    gb_cm_path = CM_DIR / f"test_{pat}_gb.png"
    nn_cm_path = CM_DIR / f"test_{pat}_nn.png"

    row_imgs = []
    for cm_path, label in [(gb_cm_path, "GradientBoosting"),
                            (nn_cm_path, "Neural Network")]:
        if cm_path.exists():
            row_imgs.append(Image(str(cm_path),
                                   width=img_w, height=img_h))
        else:
            row_imgs.append(Paragraph(f"{label}\nimage not found",
                                       small_s))

    # Patient header
    gb_auc = gb_results[pat]["auc_roc"]
    nn_auc = nn_results[pat]["auc_roc"]
    story.append(Paragraph(
        f"<b>{pat}</b> &nbsp; GB AUC: {gb_auc:.3f} &nbsp; "
        f"NN AUC: {nn_auc:.3f} &nbsp; "
        f"Grade: {grade(max(gb_auc, nn_auc))}",
        body_s))

    tbl = Table([row_imgs],
                colWidths=[img_w + 0.1*inch, img_w + 0.1*inch])
    tbl.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("GRID",    (0,0), (-1,-1), 0.3,
         colors.HexColor("#dddddd")),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.15*inch))

# ── Build ─────────────────────────────────────────────────
doc.build(story)
print(f"\n✅ PDF saved → {OUTPUT_PDF}\n")