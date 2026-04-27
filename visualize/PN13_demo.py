"""
=============================================================
  Seizure Prediction — Real-Time Demo (Multi-Channel Version)
  realtime_demo_multichannel.py

  Automatically finds the seizure with the longest prediction
  horizon, then starts the demo 1 minute before the alert fires.
  Shows all 8 EEG channels stacked vertically.

  Usage:    python3 visualize/realtime_demo_multichannel.py
  Controls: SPACE=pause  RIGHT/+=faster  LEFT/-=slower  Q=quit
=============================================================
"""

import os, re, sys, time, joblib
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from collections import deque
from pathlib import Path
from scipy import signal as scipy_signal

os.environ["TF_METAL_ENABLED"] = "1"
import mne
mne.set_log_level("WARNING")
from tensorflow import keras


# CONFIG  
PATIENT      = "PN13"   # PN05 PN07 PN10 PN12 PN13 PN14 PN16 PN17
PATIENT_AUCS = {
    "PN05":0.772,"PN07":0.856,"PN10":0.755,"PN12":0.706,
    "PN13":0.759,"PN14":0.721,"PN16":0.827,"PN17":0.774,
}
PATIENT_AUC  = PATIENT_AUCS.get(PATIENT, 0.0)

DATA_ROOT       = Path("data/siena-scalp-eeg-database-1.0.0")
MODELS_DIR      = Path("models_v6")
TARGET_CHANNELS = ["F7","T3","T5","C3","F8","T4","T6","C4"]
TARGET_SFREQ    = 250
WINDOW_SEC      = 30
STEP_SEC        = 5
PREICTAL_SEC    = 5 * 60
THRESHOLD       = 0.65
DISPLAY_CHANNEL = "T3"
DISPLAY_SEC     = 60       # 1 min rolling window (tighter for multichannel)
SPEED           = 15.0

# Demo window controls 
# How many seconds BEFORE the alert fires the demo starts
# 60  = start 1 min before alert  (tight, dramatic)
# 120 = start 2 min before alert  (more context)
# 300 = start 5 min before alert  (show lots of normal brain)
DEMO_PRE_ALERT  = 60

# Cap the prediction horizon — finds the seizure where the
# alert fires CLOSEST TO this many seconds before seizure.
# None = use the longest horizon (default)
# 120  = show ~2 min warning
# 200  = show ~3.3 min warning
# 270  = show ~4.5 min warning
MAX_HORIZON_SEC = None   # set to e.g. 120 for 2 min warning

# How many seconds of normal EEG to show BEFORE the preictal zone
# (only applies if demo starts before the preictal zone)
# This is automatically handled — just increase DEMO_PRE_ALERT
# e.g. if alert fires 4.5 min before seizure and DEMO_PRE_ALERT=300,
# demo starts 5 min before alert = 9.5 min before seizure
DEMO_POST_SEC   = 3 * 60

WINDOW_SAMPLES  = int(TARGET_SFREQ * WINDOW_SEC)
STEP_SAMPLES    = int(TARGET_SFREQ * STEP_SEC)
DISPLAY_SAMP    = int(TARGET_SFREQ * DISPLAY_SEC)
CH_SPACING      = 80

C_BG="white"; C_NORMAL="#1a9641"; C_CAUTION="#d97f00"
C_ALERT="#d7191c"; C_SEIZURE="#7b2d8b"; C_END="#228B22"
C_TEXT="#1a1a1a"; C_GRID="#e8e8e8"; C_THRESH="#d97f00"
CH_COLORS=["#2166ac","#d62728","#2ca02c","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22"]


# HELPERS
def get_ts(line):
    line=re.sub(r"(\d)\s(\d)",r"\1\2",line)
    m=re.search(r"\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b",line)
    if m: return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"

def hms(t):
    p=re.split(r"[.:\s]+",t.strip())
    return int(p[0])*3600+int(p[1])*60+int(p[2]) if len(p)>=3 else 0

def elapsed(s,e):
    d=e-s; return d+86400 if d<0 else d

def find_ch(names,targets):
    m={}
    for tgt in targets:
        for ch in names:
            c=re.sub(r"^EEG\s*","",ch).strip()
            c=re.split(r"[-–]",c)[0].strip()
            if c.upper()==tgt.upper(): m[tgt]=ch; break
    return m

def get_features(win,nch):
    fs,psd=scipy_signal.welch(win,fs=TARGET_SFREQ,nperseg=512,axis=-1)
    eps=1e-10
    BANDS={"d":(0.5,4),"t":(4,8),"a":(8,13),"b":(13,30),"g":(30,40)}
    bn=list(BANDS.keys()); bp=np.zeros((nch,5))
    for i,(_,(lo,hi)) in enumerate(BANDS.items()):
        bp[:,i]=np.log1p(psd[:,(fs>=lo)&(fs<hi)].mean(axis=1))
    ta=bp[:,bn.index("t")]/(bp[:,bn.index("a")]+eps)
    db=bp[:,bn.index("d")]/(bp[:,bn.index("b")]+eps)
    tot=psd.sum(axis=1,keepdims=True)+eps
    ent=-(np.clip(psd/tot,eps,1)*np.log(np.clip(psd/tot,eps,1))).sum(axis=1)
    return np.concatenate([bp.flatten(),ta,db,ent]).astype(np.float32)

def load_and_preprocess(edf_path,cm):
    raw=mne.io.read_raw_edf(str(edf_path),preload=True,verbose=False)
    avail=[ch for ch in cm.values() if ch in raw.ch_names]
    if len(avail)<4: return None,None,None
    raw.pick(avail)
    rd,_=raw[:]
    raw.set_eeg_reference("average",projection=False,verbose=False)
    raw.filter(0.5,40.0,method="iir",verbose=False)
    raw.notch_filter(50.0,method="iir",verbose=False)
    if raw.info["sfreq"]!=TARGET_SFREQ:
        raw.resample(TARGET_SFREQ,verbose=False)
    nd,_=raw[:]
    cnames=[re.split(r"[-–]",re.sub(r"^EEG\s*","",c).strip())[0].strip()
            for c in raw.ch_names]
    return rd.copy(),nd.copy(),cnames

def compute_baseline(norm_sig,nch,minutes=20):
    bf=[]; s=0
    bend=min(int(minutes*60*TARGET_SFREQ),norm_sig.shape[1])
    while s+WINDOW_SAMPLES<=bend and len(bf)<150:
        bf.append(get_features(norm_sig[:,s:s+WINDOW_SAMPLES],nch))
        s+=STEP_SAMPLES
    if not bf: return None,None
    bX=np.array(bf); bm=bX.mean(0); bs=bX.std(0)
    bs[bs<1e-10]=1.0
    return bm,bs

def run_model(win,nch,bm,bs,model,scaler):
    f=(get_features(win,nch)-bm)/bs
    return float(model.predict(scaler.transform(f.reshape(1,-1)),
                               verbose=0).flatten()[0])


# LOAD MODEL
print(f"\n{'='*62}\n  MULTI-CHANNEL DEMO — {PATIENT}\n{'='*62}")
model =keras.models.load_model(
    str(MODELS_DIR/f"nn_{PATIENT}.keras"),compile=False)
scaler=joblib.load(MODELS_DIR/f"scaler_{PATIENT}.pkl")
print("  Model loaded OK")

# PARSE SEIZURE LIST
pdir=DATA_ROOT/PATIENT; szs=[]
with open(list(pdir.glob("Seizures-list-*.txt"))[0]) as f:
    lines=f.readlines()
cf=None; rs=ss=se=None

def commit():
    global ss,se
    if ss and se and rs:
        on=elapsed(rs,ss); dur=elapsed(ss,se)
        if 20<dur<600: szs.append({"file":cf,"onset":on,"end":on+dur})
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

# longest prediction horizon
print(f"  Scanning {len(szs)} seizures for best prediction horizon...")

best={
    "horizon":-1,"alert_t":None,
    "sz_onset":None,"sz_end":None,
    "raw_sig":None,"norm_sig":None,
    "nch":None,"cnames":None,
    "bmean":None,"bstd":None,
}

for edf in sorted(pdir.glob("*.edf")):
    here=[s for s in szs if s["file"]==edf.name]
    if not here: continue
    try:
        cm=find_ch(
            mne.io.read_raw_edf(str(edf),preload=False,
                                verbose=False).ch_names,
            TARGET_CHANNELS)
        rd,nd,cnames=load_and_preprocess(edf,cm)
    except: continue
    if rd is None: continue
    nch_=nd.shape[0]
    bm_,bs_=compute_baseline(nd,nch_,minutes=20)
    if bm_ is None: continue

    for sz in here:
        onset_samp=int(sz["onset"]*TARGET_SFREQ)
        scan_start=max(0,onset_samp-int(5*60*TARGET_SFREQ))
        s=scan_start
        while s+WINDOW_SAMPLES<=onset_samp:
            prob=run_model(nd[:,s:s+WINDOW_SAMPLES],
                           nch_,bm_,bs_,model,scaler)
            wt=(s+WINDOW_SAMPLES)/TARGET_SFREQ
            if prob>=THRESHOLD:
                horizon=sz["onset"]-wt
                print(f"    {edf.name}  onset={sz['onset']:.0f}s  "
                      f"alert_t={wt:.0f}s  "
                      f"horizon={horizon:.0f}s ({horizon/60:.1f}min)  "
                      f"dur={sz['end']-sz['onset']:.0f}s")
                # If MAX_HORIZON_SEC set, pick closest match
                # Otherwise pick longest horizon
                if MAX_HORIZON_SEC is not None:
                    # Only consider horizons <= MAX_HORIZON_SEC
                    # Pick the one closest to target
                    if horizon <= MAX_HORIZON_SEC:
                        diff = abs(horizon - MAX_HORIZON_SEC)
                        best_diff = abs(best["horizon"] - MAX_HORIZON_SEC) \
                            if best["horizon"] > 0 else 9999
                        if diff < best_diff:
                            best.update({
                                "horizon":horizon,"alert_t":wt,
                                "sz_onset":sz["onset"],"sz_end":sz["end"],
                                "raw_sig":rd,"norm_sig":nd,
                                "nch":nch_,"cnames":cnames,
                                "bmean":bm_,"bstd":bs_,
                            })
                else:
                    if horizon>best["horizon"]:
                        best.update({
                            "horizon":horizon,"alert_t":wt,
                            "sz_onset":sz["onset"],"sz_end":sz["end"],
                            "raw_sig":rd,"norm_sig":nd,
                            "nch":nch_,"cnames":cnames,
                            "bmean":bm_,"bstd":bs_,
                        })
                break
            s+=STEP_SAMPLES

# Fallback to first seizure if none found
if best["sz_onset"] is None:
    print("  No alert found — using first seizure")
    for edf in sorted(pdir.glob("*.edf")):
        here=[s for s in szs if s["file"]==edf.name]
        if not here: continue
        try:
            cm=find_ch(
                mne.io.read_raw_edf(str(edf),preload=False,
                                    verbose=False).ch_names,
                TARGET_CHANNELS)
            rd,nd,cnames=load_and_preprocess(edf,cm)
        except: continue
        if rd is None: continue
        nch_=nd.shape[0]
        bm_,bs_=compute_baseline(nd,nch_,minutes=20)
        sz=here[0]
        best.update({
            "horizon":0,"alert_t":sz["onset"]-60,
            "sz_onset":sz["onset"],"sz_end":sz["end"],
            "raw_sig":rd,"norm_sig":nd,
            "nch":nch_,"cnames":cnames,
            "bmean":bm_,"bstd":bs_,
        })
        break

# Unpack
raw_sig    = best["raw_sig"]
norm_sig   = best["norm_sig"]
nch        = best["nch"]
ch_names_display = best["cnames"]
bmean      = best["bmean"]
bstd       = best["bstd"]
sz_onset   = best["sz_onset"]
sz_end     = best["sz_end"]
sz_dur     = sz_end-sz_onset
alert_t    = best["alert_t"]
horizon    = best["horizon"]
pre_start  = sz_onset-PREICTAL_SEC
total      = norm_sig.shape[1]

demo_start = max(0,alert_t-DEMO_PRE_ALERT)
demo_end   = min(total-WINDOW_SAMPLES,
                 int((sz_end+DEMO_POST_SEC)*TARGET_SFREQ))
ds         = int(demo_start*TARGET_SFREQ)

CH_OFFSETS = [(nch-1-i)*CH_SPACING for i in range(nch)]
Y_MIN=-CH_SPACING; Y_MAX=nch*CH_SPACING

print(f"\n  Best seizure  : onset={sz_onset:.0f}s  "
      f"end={sz_end:.0f}s  dur={sz_dur:.0f}s")
print(f"  Alert fires at: t={alert_t:.0f}s  "
      f"({horizon:.0f}s = {horizon/60:.1f} min before seizure)")
print(f"  Demo starts at: t={demo_start:.0f}s  (1 min before alert)")
print(f"  Channels: {ch_names_display}")
print(f"  Speed: {SPEED}x  |  Controls: SPACE RIGHT LEFT +/- Q\n")
time.sleep(1)


# FIGURE
plt.rcParams.update({
    "font.family":"sans-serif","figure.facecolor":C_BG,
    "axes.facecolor":C_BG,"text.color":C_TEXT,
    "axes.labelcolor":C_TEXT,"xtick.color":C_TEXT,
    "ytick.color":C_TEXT,"axes.edgecolor":"#bbbbbb",
    "grid.color":C_GRID,
})

fig=plt.figure(figsize=(14,8.5),facecolor=C_BG)
fig.canvas.manager.set_window_title(
    f"Seizure Prediction — Multi-Channel — {PATIENT}")
gs=gridspec.GridSpec(2,3,figure=fig,
                      height_ratios=[1.8,1],
                      hspace=0.45,wspace=0.32)
ax_eeg =fig.add_subplot(gs[0,:])
ax_prob=fig.add_subplot(gs[1,0:2])
ax_stat=fig.add_subplot(gs[1,2])
for ax in [ax_eeg,ax_prob,ax_stat]:
    ax.set_facecolor(C_BG)

# Multi-channel EEG
ax_eeg.set_title(f"EEG — All {nch} Channels  ({PATIENT})",
                  fontsize=12,fontweight="bold")
ax_eeg.set_xlabel("Time (seconds)",fontsize=9)
ax_eeg.set_ylim(Y_MIN,Y_MAX)
ax_eeg.grid(True,alpha=0.25,axis="x")
ax_eeg.set_yticks(CH_OFFSETS)
ax_eeg.set_yticklabels(
    ch_names_display[::-1] if len(ch_names_display)==nch
    else [f"Ch{i}" for i in range(nch)], fontsize=8)

eeg_lines=[]
for i in range(nch):
    ln,=ax_eeg.plot([],[],color=CH_COLORS[i%len(CH_COLORS)],
                    lw=0.7,alpha=0.85,zorder=3)
    eeg_lines.append(ln)
for off in CH_OFFSETS:
    ax_eeg.axhline(off-CH_SPACING//2,color="#dddddd",lw=0.5,zorder=1)

vl_alert =ax_eeg.axvline(999999,color=C_ALERT,  lw=2.5,ls="--",zorder=5)
vl_sz_on =ax_eeg.axvline(999999,color=C_SEIZURE,lw=2.5,ls="-", zorder=5)
vl_sz_end=ax_eeg.axvline(999999,color=C_END,    lw=2.5,ls="--",zorder=5)
vl_pre   =ax_eeg.axvline(999999,color="#aaaaaa",lw=1.5,ls=":", zorder=2)
lbl_alert =ax_eeg.text(0,Y_MAX*0.92,"ALERT",  color=C_ALERT,
                        fontsize=8,fontweight="bold",visible=False,zorder=6)
lbl_sz_on =ax_eeg.text(0,Y_MAX*0.92,"SEIZURE",color=C_SEIZURE,
                        fontsize=8,fontweight="bold",visible=False,zorder=6)
lbl_sz_end=ax_eeg.text(0,Y_MAX*0.92,"END",    color=C_END,
                        fontsize=8,fontweight="bold",visible=False,zorder=6)
ax_eeg.legend(handles=[
    Line2D([0],[0],color=C_ALERT,  lw=2,ls="--",label="Alert fired"),
    Line2D([0],[0],color=C_SEIZURE,lw=2,ls="-", label="Seizure onset"),
    Line2D([0],[0],color=C_END,    lw=2,ls="--",label="Seizure end"),
],fontsize=8,loc="upper right",facecolor=C_BG,edgecolor="#cccccc")

# Probability
ax_prob.set_title("Preictal Probability",fontsize=12,fontweight="bold")
ax_prob.set_xlabel("Time (seconds)",fontsize=9)
ax_prob.set_ylabel("Probability",fontsize=9)
ax_prob.set_ylim(-0.05,1.1)
ax_prob.grid(True,alpha=0.4)
ax_prob.axhline(THRESHOLD,color=C_THRESH,lw=2,ls="--",
                label=f"Alert threshold ({THRESHOLD})",zorder=3)
ax_prob.legend(fontsize=8,loc="upper left",
                facecolor=C_BG,edgecolor="#cccccc")
prob_line,=ax_prob.plot([],[],color=C_NORMAL,lw=2.5,zorder=4)
pvl_alert =ax_prob.axvline(999999,color=C_ALERT,  lw=2,ls="--",alpha=0)
pvl_sz_on =ax_prob.axvline(999999,color=C_SEIZURE,lw=2,ls="-", alpha=0)
pvl_sz_end=ax_prob.axvline(999999,color=C_END,    lw=2,ls="--",alpha=0)

# Status
ax_stat.axis("off")
ax_stat.set_title("Status",fontsize=10,fontweight="bold")
status_txt=ax_stat.text(0.5,0.72,"MONITORING",ha="center",va="center",
                         fontsize=15,fontweight="bold",color=C_NORMAL,
                         transform=ax_stat.transAxes)
sub_txt   =ax_stat.text(0.5,0.52,"",ha="center",va="center",
                         fontsize=11,color=C_TEXT,
                         transform=ax_stat.transAxes)
detail_txt=ax_stat.text(0.5,0.32,"",ha="center",va="center",
                         fontsize=9,color=C_ALERT,
                         transform=ax_stat.transAxes)
metric_txt=ax_stat.text(0.5,0.08,"",ha="center",va="center",
                         fontsize=8.5,color=C_TEXT,
                         transform=ax_stat.transAxes)
for sp in ax_stat.spines.values():
    sp.set_visible(True); sp.set_edgecolor("#cccccc")
    sp.set_linewidth(1.5)

fig.suptitle(
    f"Seizure Prediction  |  {PATIENT}  |  "
    f"All {nch} Channels  |  v6 Neural Network  |  "
    f"Threshold {THRESHOLD}",
    fontsize=10,color=C_TEXT,y=0.99)
plt.tight_layout(rect=[0,0,1,0.97])


# STATE
def make_state():
    return {
        "cur":ds,"paused":False,"speed":SPEED,
        "eeg_bufs":[deque(maxlen=DISPLAY_SAMP) for _ in range(nch)],
        "t_buf":deque(maxlen=DISPLAY_SAMP),
        "probs":[],"times":[],
        "alerted":False,"alert_fired_t":None,
        "sz_shown":False,"end_shown":False,
        "pre_drawn":False,"alert_drawn":False,
        "sz_drawn":False,"sz_end_drawn":False,
        "last_win":ds-STEP_SAMPLES,
    }

state=make_state()
i0=max(0,ds-DISPLAY_SAMP)
for ch in range(nch):
    ri=raw_sig[ch,i0:ds].copy()
    if np.max(np.abs(ri))<0.01: ri*=1e6
    state["eeg_bufs"][ch].extend(ri.tolist())
state["t_buf"].extend((np.arange(i0,ds)/TARGET_SFREQ).tolist())

def reset_state():
    s=make_state(); state.update(s)
    for vl in [vl_alert,vl_sz_on,vl_sz_end,vl_pre]:
        vl.set_xdata([999999])
    for pvl in [pvl_alert,pvl_sz_on,pvl_sz_end]:
        pvl.set_xdata([999999]); pvl.set_alpha(0)
    for lbl in [lbl_alert,lbl_sz_on,lbl_sz_end]:
        lbl.set_visible(False)
    prob_line.set_data([],[])
    for ln in eeg_lines: ln.set_data([],[])
    for ch in range(nch):
        ri=raw_sig[ch,i0:ds].copy()
        if np.max(np.abs(ri))<0.01: ri*=1e6
        state["eeg_bufs"][ch].extend(ri.tolist())
    state["t_buf"].extend((np.arange(i0,ds)/TARGET_SFREQ).tolist())

def on_key(e):
    if e.key==" ":       state["paused"]=not state["paused"]
    elif e.key=="q":     plt.close()
    elif e.key in["+","=","right"]:
        state["speed"]=min(state["speed"]*1.5,120)
        print(f"  Speed: {state['speed']:.0f}x")
    elif e.key in["-","left"]:
        state["speed"]=max(state["speed"]/1.5,0.5)
        print(f"  Speed: {state['speed']:.0f}x")

fig.canvas.mpl_connect("key_press_event",on_key)
FRAME_MS=33


# UPDATE
def update(frame):
    if state["paused"]: return
    adv=max(1,int(state["speed"]*(FRAME_MS/1000)*TARGET_SFREQ))
    ns=state["cur"]+adv
    if ns>=demo_end:
        reset_state(); return
    state["cur"]=ns
    t=ns/TARGET_SFREQ

    # Feed EEG
    e0=min(ns+adv,total)
    for ch in range(nch):
        rn=raw_sig[ch,ns:e0].copy()
        if np.max(np.abs(rn))<0.01: rn*=1e6
        state["eeg_bufs"][ch].extend(rn.tolist())
    state["t_buf"].extend((np.arange(ns,e0)/TARGET_SFREQ).tolist())

    # Run model
    while (state["cur"]-state["last_win"])>=STEP_SAMPLES:
        ws=state["last_win"]+STEP_SAMPLES
        we=ws+WINDOW_SAMPLES
        state["last_win"]=ws
        if we>total: break
        prob=run_model(norm_sig[:,ws:we],nch,bmean,bstd,model,scaler)
        wt=we/TARGET_SFREQ
        state["probs"].append(prob)
        state["times"].append(wt)
        if prob>=THRESHOLD and not state["alerted"]:
            state["alerted"]=True
            state["alert_fired_t"]=t
            h=sz_onset-wt
            print(f"  [ALERT] t={wt:.1f}s  prob={prob:.3f}  "
                  f"horizon={h:.0f}s ({h/60:.1f}min)")
        if t>=sz_onset and not state["sz_shown"]:
            state["sz_shown"]=True
            print(f"  [SEIZURE] t={sz_onset:.1f}s")
        if t>=sz_end and not state["end_shown"]:
            state["end_shown"]=True
            print(f"  [END] t={sz_end:.1f}s")

    # EEG channels
    ta=np.array(state["t_buf"])
    if len(ta)>1:
        ax_eeg.set_xlim(ta[-1]-DISPLAY_SEC,ta[-1])
        for ch in range(nch):
            ea=np.array(state["eeg_bufs"][ch])
            std=ea.std() if ea.std()>0 else 1
            ea_off=(ea/(std*3))*(CH_SPACING*0.45)+CH_OFFSETS[ch]
            eeg_lines[ch].set_data(ta,ea_off)

        if t>=pre_start and not state["pre_drawn"]:
            vl_pre.set_xdata([pre_start]); state["pre_drawn"]=True

        if state["alerted"] and not state["alert_drawn"]:
            vl_alert.set_xdata([state["alert_fired_t"]])
            pvl_alert.set_xdata([state["alert_fired_t"]])
            pvl_alert.set_alpha(0.9)
            lbl_alert.set_position((state["alert_fired_t"]+2,Y_MAX*0.88))
            lbl_alert.set_visible(True)
            state["alert_drawn"]=True

        if t>=sz_onset and not state["sz_drawn"]:
            vl_sz_on.set_xdata([sz_onset])
            pvl_sz_on.set_xdata([sz_onset])
            pvl_sz_on.set_alpha(0.9)
            lbl_sz_on.set_position((sz_onset+2,Y_MAX*0.88))
            lbl_sz_on.set_visible(True)
            state["sz_drawn"]=True

        if t>=sz_end and not state["sz_end_drawn"]:
            vl_sz_end.set_xdata([sz_end])
            pvl_sz_end.set_xdata([sz_end])
            pvl_sz_end.set_alpha(0.9)
            lbl_sz_end.set_position((sz_end+2,Y_MAX*0.88))
            lbl_sz_end.set_visible(True)
            state["sz_end_drawn"]=True

    # Probability
    if state["probs"]:
        ph=np.array(state["probs"]); th=np.array(state["times"])
        prob_line.set_data(th,ph)
        cp=ph[-1]
        col=(C_ALERT if cp>=THRESHOLD else
             C_CAUTION if cp>=0.4 else C_NORMAL)
        prob_line.set_color(col)
        ax_prob.set_xlim(max(th[0],th[-1]-DISPLAY_SEC*3),th[-1]+10)

    # Status
    cp=state["probs"][-1] if state["probs"] else 0.0
    pct=int(cp*100); spd=f"{state['speed']:.0f}x"
    col_m=(C_ALERT if cp>=THRESHOLD else
           C_CAUTION if cp>=0.4 else C_NORMAL)

    if t>=sz_end:
        status_txt.set_text("MONITORING"); status_txt.set_color(C_NORMAL)
        sub_txt.set_text("Seizure ended\nBack to monitoring")
        sub_txt.set_color(C_NORMAL); detail_txt.set_text("")
        ax_stat.set_facecolor(C_BG)
    elif t>=sz_onset:
        status_txt.set_text("SEIZURE"); status_txt.set_color(C_SEIZURE)
        sub_txt.set_text("Seizure in progress")
        sub_txt.set_color(C_SEIZURE)
        detail_txt.set_text(
            f"Predicted {horizon/60:.1f} min early" if horizon>0 else "")
        detail_txt.set_color(C_SEIZURE)
        ax_stat.set_facecolor("#f8f0ff")
    elif state["alerted"]:
        el=max(0,t-state["alert_fired_t"])
        am=int(el//60); as2=int(el%60)
        status_txt.set_text(">>> ALERT <<<"); status_txt.set_color(C_ALERT)
        sub_txt.set_text(f"Alert active: {am}m {as2:02d}s")
        sub_txt.set_color(C_ALERT)
        detail_txt.set_text(
            f"Model crossed {int(THRESHOLD*100)}% threshold\n")
        detail_txt.set_color(C_ALERT)
        ax_stat.set_facecolor("#fff5f5")
    else:
        status_txt.set_text("MONITORING"); status_txt.set_color(C_NORMAL)
        sub_txt.set_text(""); detail_txt.set_text("")
        ax_stat.set_facecolor(C_BG)

    metric_txt.set_text(
        f"Certainty: {pct}%   Threshold: {int(THRESHOLD*100)}%\n"
        f"AUC: {PATIENT_AUC} ({PATIENT})   Speed: {spd}")
    metric_txt.set_color(col_m)
    fig.canvas.draw_idle()


# RUN
ani=FuncAnimation(fig,update,interval=FRAME_MS,
                   cache_frame_data=False,blit=False)
plt.show()