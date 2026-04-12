import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")


def extract_first_timestamp(line: str):
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    match = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None

def hms_to_sec(t: str) -> float:
    parts = re.split(r"[.:\s]+", t.strip())
    if len(parts) >= 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0

def duration_sec(start: float, end: float) -> float:
    d = end - start
    return d + 24 * 3600 if d < 0 else d

def parse_seizure_file(txt_path: Path) -> list:
    patient = txt_path.parent.name
    seizures = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    current_file = None
    rec_start = rec_end = sz_start = sz_end = None

    def try_commit():
        nonlocal sz_start, sz_end
        if sz_start is not None and sz_end is not None and rec_start is not None:
            rec_dur = duration_sec(rec_start, rec_end) if rec_end else 0
            offset  = duration_sec(rec_start, sz_start)
            sz_dur  = duration_sec(sz_start, sz_end)
            if 20 < sz_dur < 600:
                seizures.append({
                    "patient":          patient,
                    "file":             current_file or "?",
                    "rec_dur_min":      round(rec_dur / 60, 1),
                    "sz_start_in_file": round(offset / 60, 2),
                    "sz_end_in_file":   round((offset + sz_dur) / 60, 2),
                    "sz_duration_sec":  round(sz_dur, 1),
                })
        sz_start = None
        sz_end   = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        ll = line.lower()
        if re.match(r"seizure\s+n\s+\d+", ll):
            try_commit()
        elif "file name" in ll:
            p = re.split(r":\s*", line, maxsplit=1)
            if len(p) > 1:
                current_file = p[1].strip()
        elif "registration start" in ll:
            t = extract_first_timestamp(line)
            if t: rec_start = hms_to_sec(t)
        elif "registration end" in ll:
            t = extract_first_timestamp(line)
            if t: rec_end = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?start\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_start = hms_to_sec(t)
        elif re.search(r"(seizure\s+)?end\s+time", ll):
            t = extract_first_timestamp(line)
            if t: sz_end = hms_to_sec(t)
    try_commit()
    return seizures


# Load data 
subject_info = pd.read_csv(DATA_ROOT / "subject_info.csv")
subject_info.columns = subject_info.columns.str.strip()

all_seizures = []
for pdir in sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("PN")]):
    for txt in pdir.glob("Seizures-list-*.txt"):
        all_seizures.extend(parse_seizure_file(txt))
seizure_df = pd.DataFrame(all_seizures)

total_rec_min = subject_info["rec_time_minutes"].sum()
total_sz_min  = seizure_df["sz_duration_sec"].sum() / 60

# Figure: two side-by-side plots 
fig, (ax_left, ax_right) = plt.subplots(
    1, 2,
    figsize=(22, 8),
    gridspec_kw={"width_ratios": [1, 3]}
)
fig.suptitle("Siena Scalp EEG — Class Imbalance & Seizure Timeline",
             fontsize=15, fontweight="bold")

# LEFT: Class imbalance 
categories = ["Non-seizure", "Seizure"]
values     = [total_rec_min - total_sz_min, total_sz_min]
bars = ax_left.bar(categories, values,
                   color=["steelblue", "tomato"],
                   edgecolor="white", width=0.5)
ax_left.set_ylabel("Minutes", fontsize=11)
ax_left.set_title("Class Imbalance\n(Total Dataset)", fontweight="bold", fontsize=12)
ax_left.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, values):
    ax_left.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{val:.0f} min\n({val/60:.1f} hrs)" if val > 100 else f"{val:.1f} min",
        ha="center", fontsize=10, fontweight="bold"
    )
ratio = (total_rec_min - total_sz_min) / total_sz_min
ax_left.text(0.5, 0.5,
             f"~{ratio:.0f}:1\nimbalance ratio",
             transform=ax_left.transAxes,
             ha="center", va="center",
             fontsize=13, color="tomato", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="tomato", alpha=0.8))

# RIGHT: Seizure timeline 
unique_patients = sorted(seizure_df["patient"].unique())
patient_y       = {p: i for i, p in enumerate(unique_patients)}
bar_height      = 0.55

for _, row in seizure_df.iterrows():
    y = patient_y[row["patient"]]
    # Recording bar
    ax_right.barh(y, row["rec_dur_min"], left=0, height=bar_height,
                  color="lightsteelblue", alpha=0.6, zorder=2)
    # Seizure marker — make it wider so it's visible (min width 3 min for display)
    sz_width = max(row["sz_end_in_file"] - row["sz_start_in_file"], 3)
    ax_right.barh(y, sz_width, left=row["sz_start_in_file"],
                  height=bar_height, color="tomato", alpha=0.9, zorder=3)

# Patient labels with seizure count
ytick_labels = []
for p in unique_patients:
    n = len(seizure_df[seizure_df["patient"] == p])
    ytick_labels.append(f"{p}  ({n} sz)")

ax_right.set_yticks(list(patient_y.values()))
ax_right.set_yticklabels(ytick_labels, fontsize=10)
ax_right.set_xlabel("Time into recording (minutes)", fontsize=11)
ax_right.set_title("Seizure Locations Within Each Recording File",
                   fontweight="bold", fontsize=12)
ax_right.spines[["top", "right"]].set_visible(False)
ax_right.set_xlim(left=0)

# Add subtle grid
ax_right.xaxis.grid(True, alpha=0.3, linestyle="--")
ax_right.set_axisbelow(True)

rec_patch = mpatches.Patch(color="lightsteelblue", alpha=0.7, label="Recording window")
sz_patch  = mpatches.Patch(color="tomato", alpha=0.9, label="Seizure (width exaggerated for visibility)")
ax_right.legend(handles=[rec_patch, sz_patch],
                loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout(pad=2.0)
plt.savefig("timeline_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to timeline_chart.png")