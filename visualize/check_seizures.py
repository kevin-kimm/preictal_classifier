"""Quick script to check seizure durations for all patients."""
import re
from pathlib import Path

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")

GOOD_PATIENTS = {
    "PN05": 0.772, "PN07": 0.856, "PN10": 0.755,
    "PN12": 0.706, "PN13": 0.759, "PN14": 0.721,
    "PN16": 0.827, "PN17": 0.774
}

def get_ts(line):
    line = re.sub(r"(\d)\s(\d)", r"\1\2", line)
    m = re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b", line)
    if m: return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"

def hms(t):
    p = re.split(r"[.:\s]+", t.strip())
    return int(p[0])*3600+int(p[1])*60+int(p[2]) if len(p)>=3 else 0

def elapsed(s, e):
    d = e-s; return d+86400 if d<0 else d

print(f"\n{'='*55}")
print(f"  Seizure durations for predictable patients (AUC >= 0.70)")
print(f"{'='*55}")
print(f"  {'Patient':<8} {'AUC':>6}  Seizures (duration)")
print(f"  {'-'*50}")

for patient, auc in sorted(GOOD_PATIENTS.items(),
                             key=lambda x: -x[1]):
    pdir = DATA_ROOT / patient
    txt  = list(pdir.glob("Seizures-list-*.txt"))
    if not txt: continue
    with open(txt[0]) as f: lines = f.readlines()

    seizures = []
    cf=None; rs=ss=se=None

    def commit():
        global ss, se
        if ss and se and rs:
            on=elapsed(rs,ss); dur=elapsed(ss,se)
            if 20<dur<600:
                seizures.append({"dur":dur,"onset":on})
        ss=None; se=None

    for raw in lines:
        l=raw.strip(); ll=l.lower()
        if not l: continue
        if re.match(r"seizure\s+n\s+\d+",ll): commit()
        elif "registration start" in ll:
            t=get_ts(l)
            if t: rs=hms(t)
        elif re.search(r"(seizure\s+)?start\s+time",ll):
            t=get_ts(l)
            if t: ss=hms(t)
        elif re.search(r"(seizure\s+)?end\s+time",ll):
            t=get_ts(l)
            if t: se=hms(t)
    commit()

    if seizures:
        durs = [f"{s['dur']:.0f}s" for s in seizures]
        best = max(s['dur'] for s in seizures)
        print(f"  {patient:<8} {auc:>6.3f}  "
              f"{' | '.join(durs)}  "
              f"(longest: {best:.0f}s)")

print(f"{'='*55}\n")