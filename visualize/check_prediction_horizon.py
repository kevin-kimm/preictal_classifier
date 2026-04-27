"""
Check prediction horizon for all predictable patients.
Loads each patient's best model and finds how early
the alert fires before seizure onset.
"""

import os, re, json
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal

os.environ["TF_METAL_ENABLED"] = "1"
import mne
mne.set_log_level("WARNING")
import joblib
from tensorflow import keras

DATA_ROOT = Path("data/siena-scalp-eeg-database-1.0.0")
MODELS_DIR = Path("models_v6")
TARGET_CHANNELS = ["F7","T3","T5","C3","F8","T4","T6","C4"]
TARGET_SFREQ = 250
WINDOW_SEC   = 30
STEP_SEC     = 5
THRESHOLD    = 0.65
WINDOW_SAMPLES = int(TARGET_SFREQ * WINDOW_SEC)
STEP_SAMPLES   = int(TARGET_SFREQ * STEP_SEC)

GOOD_PATIENTS = {
    "PN05": 0.772, "PN07": 0.856, "PN10": 0.755,
    "PN12": 0.706, "PN13": 0.759, "PN14": 0.721,
    "PN16": 0.827, "PN17": 0.774,
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

def find_ch(names, targets):
    m = {}
    for tgt in targets:
        for ch in names:
            c = re.sub(r"^EEG\s*","",ch).strip()
            c = re.split(r"[-–]",c)[0].strip()
            if c.upper()==tgt.upper(): m[tgt]=ch; break
    return m

def get_features(win, nch):
    fs, psd = scipy_signal.welch(win, fs=TARGET_SFREQ,
                                  nperseg=512, axis=-1)
    eps = 1e-10
    BANDS = {"d":(0.5,4),"t":(4,8),"a":(8,13),
             "b":(13,30),"g":(30,40)}
    bn = list(BANDS.keys())
    bp = np.zeros((nch,5))
    for i,(_,(lo,hi)) in enumerate(BANDS.items()):
        bp[:,i] = np.log1p(psd[:,(fs>=lo)&(fs<hi)].mean(axis=1))
    ta  = bp[:,bn.index("t")]/(bp[:,bn.index("a")]+eps)
    db  = bp[:,bn.index("d")]/(bp[:,bn.index("b")]+eps)
    tot = psd.sum(axis=1,keepdims=True)+eps
    ent = -(np.clip(psd/tot,eps,1)*np.log(np.clip(psd/tot,eps,1))).sum(axis=1)
    return np.concatenate([bp.flatten(),ta,db,ent]).astype(np.float32)

print(f"\n{'='*62}")
print(f"  PREDICTION HORIZON — how early does each patient get warned?")
print(f"{'='*62}")
print(f"  {'Patient':<8} {'AUC':>6}  {'Seizure':>8}  "
      f"{'Alert at':>10}  {'Horizon':>10}  {'Sz dur':>7}")
print(f"  {'-'*58}")

results = []

for patient, auc in sorted(GOOD_PATIENTS.items(),
                             key=lambda x: -x[1]):
    pdir  = DATA_ROOT / patient
    txt   = list(pdir.glob("Seizures-list-*.txt"))
    if not txt: continue

    # Parse seizures
    szs = []
    with open(txt[0]) as f: lines = f.readlines()
    cf=None; rs=ss=se=None

    def commit():
        global ss, se
        if ss and se and rs:
            on=elapsed(rs,ss); dur=elapsed(ss,se)
            if 20<dur<600:
                szs.append({"file":cf,"onset":on,"end":on+dur,"dur":dur})
        ss=None; se=None

    for raw in lines:
        l=raw.strip(); ll=l.lower()
        if not l: continue
        if re.match(r"seizure\s+n\s+\d+",ll): commit()
        elif "file name" in ll:
            p=re.split(r":\s*",l,maxsplit=1)
            if len(p)>1: cf=p[1].strip()
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

    if not szs: continue

    # Load model
    nn_path = MODELS_DIR / f"nn_{patient}.keras"
    sc_path = MODELS_DIR / f"scaler_{patient}.pkl"
    if not nn_path.exists(): continue
    model  = keras.models.load_model(str(nn_path), compile=False)
    scaler = joblib.load(sc_path)

    # Check each seizure
    best_horizon = -999
    best_sz      = None

    for sz in szs:
        edf_path = pdir / sz["file"]
        if not edf_path.exists(): continue
        try:
            raw = mne.io.read_raw_edf(str(edf_path),
                                       preload=True, verbose=False)
        except: continue
        cm    = find_ch(raw.ch_names, TARGET_CHANNELS)
        avail = [ch for ch in cm.values() if ch in raw.ch_names]
        if len(avail)<4: continue
        raw.pick(avail)
        raw.set_eeg_reference("average",projection=False,verbose=False)
        raw.filter(0.5,40.0,method="iir",verbose=False)
        raw.notch_filter(50.0,method="iir",verbose=False)
        if raw.info["sfreq"]!=TARGET_SFREQ:
            raw.resample(TARGET_SFREQ,verbose=False)
        data,_ = raw[:]
        nch    = data.shape[0]
        total  = data.shape[1]

        # Baseline from first 20 min
        bfeats = []
        s = 0; bend = min(int(20*60*TARGET_SFREQ), total)
        while s+WINDOW_SAMPLES<=bend and len(bfeats)<150:
            bfeats.append(get_features(data[:,s:s+WINDOW_SAMPLES], nch))
            s += STEP_SAMPLES
        if not bfeats: continue
        bX = np.array(bfeats)
        bmean = bX.mean(0); bstd = bX.std(0)
        bstd[bstd<1e-10] = 1.0

        # Scan preictal zone — 5 min window before seizure
        onset_samp = int(sz["onset"] * TARGET_SFREQ)
        scan_start = max(0, onset_samp - int(5*60*TARGET_SFREQ))

        first_alert = None
        s = scan_start
        while s + WINDOW_SAMPLES <= onset_samp:
            win  = data[:, s:s+WINDOW_SAMPLES]
            f    = get_features(win, nch)
            fn   = (f - bmean) / bstd
            fsc  = scaler.transform(fn.reshape(1,-1))
            prob = float(model.predict(fsc, verbose=0).flatten()[0])
            wt   = (s + WINDOW_SAMPLES) / TARGET_SFREQ
            if prob >= THRESHOLD:
                first_alert = wt
                break
            s += STEP_SAMPLES

        if first_alert is not None:
            horizon = sz["onset"] - first_alert
            if horizon > best_horizon:
                best_horizon = horizon
                best_sz = sz

    if best_sz and best_horizon > 0:
        results.append({
            "patient":  patient,
            "auc":      auc,
            "horizon":  best_horizon,
            "sz_dur":   best_sz["dur"],
            "sz_onset": best_sz["onset"],
        })
        print(f"  {patient:<8} {auc:>6.3f}  "
              f"{best_sz['onset']:>8.0f}s  "
              f"{best_sz['onset']-best_horizon:>10.0f}s  "
              f"{best_horizon:>8.0f}s ({best_horizon/60:.1f}min)  "
              f"{best_sz['dur']:>5.0f}s")
    else:
        print(f"  {patient:<8} {auc:>6.3f}  "
              f"no alert found in preictal window")

print(f"  {'-'*58}")
if results:
    best = max(results, key=lambda x: x["horizon"])
    print(f"\n  Best prediction horizon: {best['patient']} — "
          f"{best['horizon']:.0f}s ({best['horizon']/60:.1f} min) early")
    print(f"  Recommended for demo   : {best['patient']}")
print(f"{'='*62}\n")