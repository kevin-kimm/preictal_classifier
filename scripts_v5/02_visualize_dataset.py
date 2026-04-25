import re
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

mne.set_log_level("WARNING")

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")


# Extract the FIRST valid HH.MM.SS or HH:MM:SS
# timestamp from a line, ignoring anything after it
def extract_first_timestamp(line: str):
    # Remove spaces that are typos inside timestamps e.g. "1 6.49.25"
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    match = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return None


# Convert HH.MM.SS to total seconds

def hms_to_sec(t: str) -> float:
    parts = re.split(r"[.:\s]+", t.strip())
    if len(parts) >= 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


# Handle midnight rollover
def duration_sec(start_sec: float, end_sec: float) -> float:
    dur = end_sec - start_sec
    if dur < 0:
        dur += 24 * 3600
    return dur



# MAIN PARSER
def parse_seizure_file(txt_path: Path) -> list:
    patient = txt_path.parent.name
    seizures = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    current_file = None
    rec_start    = None
    rec_end      = None
    sz_start     = None
    sz_end       = None

    def try_commit():
        nonlocal sz_start, sz_end, rec_start, rec_end
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
        line_lower = line.lower()

        if re.match(r"seizure\s+n\s+\d+", line_lower):
            try_commit()
            continue

        if "file name" in line_lower:
            parts = re.split(r":\s*", line, maxsplit=1)
            if len(parts) > 1:
                current_file = parts[1].strip()
            continue

        if "registration start" in line_lower:
            t = extract_first_timestamp(line)
            if t:
                rec_start = hms_to_sec(t)
            continue

        if "registration end" in line_lower:
            t = extract_first_timestamp(line)
            if t:
                rec_end = hms_to_sec(t)
            continue

        if re.search(r"(seizure\s+)?start\s+time", line_lower):
            t = extract_first_timestamp(line)
            if t:
                sz_start = hms_to_sec(t)
            continue

        if re.search(r"(seizure\s+)?end\s+time", line_lower):
            t = extract_first_timestamp(line)
            if t:
                sz_end = hms_to_sec(t)
            continue

    try_commit()
    return seizures


# STEP 1: Load all data
print("\nLoading dataset\n")

subject_info = pd.read_csv(DATA_ROOT / "subject_info.csv")
subject_info.columns = subject_info.columns.str.strip()

all_seizures = []
patient_dirs = sorted([d for d in DATA_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("PN")])

parse_report = []
for pdir in patient_dirs:
    txt_files = list(pdir.glob("Seizures-list-*.txt"))
    if txt_files:
        found = parse_seizure_file(txt_files[0])
        all_seizures.extend(found)
        expected = subject_info.loc[
            subject_info["patient_id"] == pdir.name, "number_seizures"
        ].values
        expected = int(expected[0]) if len(expected) else "?"
        status = "Good" if len(found) == expected else "Fault"
        parse_report.append((pdir.name, len(found), expected, status))
    else:
        parse_report.append((pdir.name, 0, "?", "X"))

seizure_df = pd.DataFrame(all_seizures)


# STEP 2: Print tables
print("=" * 70)
print("  TABLE 1 — PATIENT OVERVIEW")
print("=" * 70)
print(subject_info.to_string(index=False))

print("\n" + "=" * 70)
print("  TABLE 2 — PARSER ACCURACY CHECK")
print("=" * 70)
print(f"  {'Patient':<10} {'Found':<10} {'Expected':<10} Status")
print(f"  {'-'*45}")
for patient, found, expected, status in parse_report:
    print(f"  {patient:<10} {str(found):<10} {str(expected):<10} {status}")

print("\n" + "=" * 70)
print("  TABLE 3 — ALL SEIZURES")
print("=" * 70)
print(f"{'Patient':<10} {'File':<18} {'Rec (min)':<12} "
      f"{'Sz Start (min)':<16} {'Sz End (min)':<14} {'Duration (s)'}")
print("-" * 75)
for _, row in seizure_df.iterrows():
    print(f"{row['patient']:<10} {row['file']:<18} {row['rec_dur_min']:<12} "
          f"{row['sz_start_in_file']:<16} {row['sz_end_in_file']:<14} "
          f"{row['sz_duration_sec']}")

print("\n" + "=" * 70)
print("  TABLE 4 — PER-PATIENT SUMMARY")
print("=" * 70)
summary = seizure_df.groupby("patient").agg(
    num_seizures      = ("sz_duration_sec", "count"),
    avg_duration_sec  = ("sz_duration_sec", "mean"),
    min_duration_sec  = ("sz_duration_sec", "min"),
    max_duration_sec  = ("sz_duration_sec", "max"),
    total_seizure_sec = ("sz_duration_sec", "sum"),
).reset_index()
summary["avg_duration_sec"]  = summary["avg_duration_sec"].round(1)
summary["total_seizure_sec"] = summary["total_seizure_sec"].round(1)
print(summary.to_string(index=False))

total_rec_min = subject_info["rec_time_minutes"].sum()
total_sz_sec  = seizure_df["sz_duration_sec"].sum()
total_sz_min  = total_sz_sec / 60
ratio         = (total_rec_min - total_sz_min) / total_sz_min

print(f"\n{'=' * 70}")
print(f"  CLASS IMBALANCE SUMMARY")
print(f"{'=' * 70}")
print(f"  Total recording time : {total_rec_min:.0f} minutes ({total_rec_min/60:.1f} hours)")
print(f"  Total seizure time   : {total_sz_min:.1f} minutes ({total_sz_sec:.0f} seconds)")
print(f"  Non-seizure time     : {total_rec_min - total_sz_min:.0f} minutes")
print(f"  Imbalance ratio      : ~{ratio:.0f}:1  (non-seizure : seizure)")
print(f"  Never use accuracy as your metric — use F1 / recall / AUC\n")


# STEP 3: Visualizations
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Siena Scalp EEG Dataset — Full Overview",
             fontsize=16, fontweight="bold", y=0.98)
colors = plt.cm.tab20.colors

ax1 = fig.add_subplot(3, 3, 1)
patients  = subject_info["patient_id"]
rec_times = subject_info["rec_time_minutes"]
bars = ax1.barh(patients, rec_times, color=colors[:len(patients)])
ax1.set_xlabel("Minutes")
ax1.set_title("Total Recording Time per Patient", fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, rec_times):
    ax1.text(val + 5, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}m", va="center", fontsize=7)

ax2 = fig.add_subplot(3, 3, 2)
num_sz = subject_info["number_seizures"]
bars2  = ax2.barh(patients, num_sz, color=colors[:len(patients)])
ax2.set_xlabel("Count")
ax2.set_title("Number of Seizures per Patient", fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars2, num_sz):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", fontsize=8)

ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(seizure_df["sz_duration_sec"], bins=20,
         color="steelblue", edgecolor="white")
mean_dur = seizure_df["sz_duration_sec"].mean()
ax3.axvline(mean_dur, color="red", linestyle="--",
            label=f"Mean: {mean_dur:.0f}s")
ax3.set_xlabel("Duration (seconds)")
ax3.set_ylabel("Count")
ax3.set_title("Seizure Duration Distribution", fontweight="bold")
ax3.legend(fontsize=8)
ax3.spines[["top", "right"]].set_visible(False)

ax4 = fig.add_subplot(3, 3, 4)
ax4.hist(subject_info["age_years"], bins=10, color="coral", edgecolor="white")
ax4.set_xlabel("Age (years)")
ax4.set_ylabel("Count")
ax4.set_title("Patient Age Distribution", fontweight="bold")
ax4.spines[["top", "right"]].set_visible(False)

ax5 = fig.add_subplot(3, 3, 5)
gender_counts = subject_info["gender"].value_counts()
ax5.pie(gender_counts, labels=gender_counts.index,
        autopct="%1.0f%%", colors=["steelblue", "coral"],
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax5.set_title("Gender Distribution", fontweight="bold")

ax6 = fig.add_subplot(3, 3, 6)
sz_type_counts = subject_info["seizure"].value_counts()
ax6.pie(sz_type_counts, labels=sz_type_counts.index,
        autopct="%1.0f%%",
        colors=["steelblue", "coral", "mediumseagreen"],
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax6.set_title("Seizure Type Distribution\n(IAS / FBTC / WIAS)", fontweight="bold")

ax7 = fig.add_subplot(3, 3, 7)
categories = ["Non-seizure", "Seizure"]
values     = [total_rec_min - total_sz_min, total_sz_min]
bars7      = ax7.bar(categories, values,
                     color=["steelblue", "tomato"],
                     edgecolor="white", width=0.5)
ax7.set_ylabel("Minutes")
ax7.set_title("Class Imbalance (Total Dataset)", fontweight="bold")
ax7.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars7, values):
    ax7.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             f"{val:.0f} min", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("dataset_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to dataset_overview.png")