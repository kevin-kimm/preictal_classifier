"""
=============================================================
  Seizure Prediction — Multi-Seizure Compilation Demo
  multi_seizure_demo.py

  Splices together up to 5 seizures from PN10 (5 seizures!)
  into a single continuous 10-min demo. Watch the algorithm
  detect each seizure in real time, one after another.

  Between seizures: brief "TIME SKIP" transition screen.

  Usage:    python3 visualize/multi_seizure_demo.py
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
PATIENT         = "PN10"   # has 5 seizures 
PATIENT_AUC     = 0.755
MAX_SEIZURES    = 5        # how many seizures to include
PRE_SEIZURE_SEC = 90       # seconds before alert to show per seizure
POST_SEIZURE_SEC= 45       # seconds after seizure end to show
DATA_ROOT       = Path("data/siena-scalp-eeg-database-1.0.0")
MODELS_DIR      = Path("models_v6")
TARGET_CHANNELS = ["F7","T3","T5","C3","F8","T4","T6","C4"]
DISPLAY_CHANNEL = "F7" "F8"
TARGET_SFREQ    = 250
WINDOW_SEC      = 30
STEP_SEC        = 5
PREICTAL_SEC    = 5 * 60
THRESHOLD       = 0.65
DISPLAY_SEC     = 90
SPEED           = 10.0
CH_SPACING      = 80

WINDOW_SAMPLES  = int(TARGET_SFREQ * WINDOW_SEC)
STEP_SAMPLES    = int(TARGET_SFREQ * STEP_SEC)
DISPLAY_SAMP    = int(TARGET_SFREQ * DISPLAY_SEC)

C_BG="white"; C_NORMAL="#1a9641"; C_CAUTION="#d97f00"
C_ALERT="#d7191c"; C_SEIZURE="#7b2d8b"; C_END="#228B22"
C_TEXT="#1a1a1a"; C_GRID="#e8e8e8"; C_THRESH="#d97f00"


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

def run_model(win,nch,bm,bs,mdl,scl):
    f=(get_features(win,nch)-bm)/bs
    return float(mdl.predict(scl.transform(f.reshape(1,-1)),
                              verbose=0).flatten()[0])


# LOAD MODEL + ALL SEIZURES
print(f"\n{'='*62}\n  MULTI-SEIZURE COMPILATION — {PATIENT}\n{'='*62}")
model =keras.models.load_model(
    str(MODELS_DIR/f"nn_{PATIENT}.keras"),compile=False)
scaler=joblib.load(MODELS_DIR/f"scaler_{PATIENT}.pkl")
print("  Model loaded OK")

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

print(f"  Found {len(szs)} seizures total")

# BUILD SEGMENTS — one per seizure
# Each segment = [pre_alert ... alert ... seizure ... post]
print("  Building seizure segments...")
segments = []   # list of dicts, one per seizure

for edf in sorted(pdir.glob("*.edf")):
    if len(segments) >= MAX_SEIZURES: break
    here=[s for s in szs if s["file"]==edf.name]
    if not here: continue
    try:
        raw_=mne.io.read_raw_edf(str(edf),preload=True,verbose=False)
    except: continue
    cm=find_ch(raw_.ch_names,TARGET_CHANNELS)
    avail=[ch for ch in cm.values() if ch in raw_.ch_names]
    if len(avail)<4: continue
    raw_.pick(avail)
    rd,_=raw_[:]
    raw_.set_eeg_reference("average",projection=False,verbose=False)
    raw_.filter(0.5,40.0,method="iir",verbose=False)
    raw_.notch_filter(50.0,method="iir",verbose=False)
    if raw_.info["sfreq"]!=TARGET_SFREQ:
        raw_.resample(TARGET_SFREQ,verbose=False)
    nd,_=raw_[:]
    nch_=nd.shape[0]
    cnames_=[re.split(r"[-–]",re.sub(r"^EEG\s*","",c).strip())[0].strip()
             for c in raw_.ch_names]
    dci_=next((i for i,c in enumerate(cnames_)
                if c.upper()==DISPLAY_CHANNEL.upper()),0)

    # Baseline from first 20 min
    bf=[]; s=0; bend=min(int(20*60*TARGET_SFREQ),nd.shape[1])
    while s+WINDOW_SAMPLES<=bend and len(bf)<100:
        bf.append(get_features(nd[:,s:s+WINDOW_SAMPLES],nch_))
        s+=STEP_SAMPLES
    if not bf: continue
    bX=np.array(bf); bm=bX.mean(0); bs=bX.std(0); bs[bs<1e-10]=1.0

    for sz in here:
        if len(segments)>=MAX_SEIZURES: break
        onset_samp=int(sz["onset"]*TARGET_SFREQ)
        end_samp  =int(sz["end"]*TARGET_SFREQ)
        scan_start=max(0,onset_samp-int(5*60*TARGET_SFREQ))

        # Find alert time
        alert_samp=None
        s=scan_start
        while s+WINDOW_SAMPLES<=onset_samp:
            prob=run_model(nd[:,s:s+WINDOW_SAMPLES],nch_,bm,bs,model,scaler)
            if prob>=THRESHOLD:
                alert_samp=s+WINDOW_SAMPLES
                break
            s+=STEP_SAMPLES

        if alert_samp is None:
            print(f"    {edf.name} onset={sz['onset']:.0f}s  "
                  f"no alert — skipping")
            continue

        horizon=(onset_samp-alert_samp)/TARGET_SFREQ
        seg_start=max(0,alert_samp-int(PRE_SEIZURE_SEC*TARGET_SFREQ))
        seg_end  =min(nd.shape[1],
                      end_samp+int(POST_SEIZURE_SEC*TARGET_SFREQ))

        segments.append({
            "raw_seg":   rd[:,seg_start:seg_end].copy(),
            "norm_seg":  nd[:,seg_start:seg_end].copy(),
            "nch":       nch_,
            "dci":       dci_,
            "bmean":     bm.copy(),
            "bstd":      bs.copy(),
            # Times relative to segment start
            "alert_sec": (alert_samp-seg_start)/TARGET_SFREQ,
            "onset_sec": (onset_samp-seg_start)/TARGET_SFREQ,
            "end_sec":   (end_samp-seg_start)/TARGET_SFREQ,
            "horizon":   horizon,
            "dur":       sz["end"]-sz["onset"],
            "sz_num":    len(segments)+1,
        })
        print(f"    Seizure {len(segments)}: {edf.name}  "
              f"onset={sz['onset']:.0f}s  "
              f"horizon={horizon:.0f}s ({horizon/60:.1f}min)  "
              f"dur={sz['end']-sz['onset']:.0f}s")

if not segments:
    print("No predictable seizures found!"); sys.exit(1)

print(f"\n  Loaded {len(segments)} seizures for compilation")
nch   = segments[0]["nch"]
dci   = segments[0]["dci"]
time.sleep(1)


# FIGURE
plt.rcParams.update({
    "font.family":"sans-serif","figure.facecolor":C_BG,
    "axes.facecolor":C_BG,"text.color":C_TEXT,
    "axes.labelcolor":C_TEXT,"xtick.color":C_TEXT,
    "ytick.color":C_TEXT,"axes.edgecolor":"#bbbbbb","grid.color":C_GRID,
})
fig=plt.figure(figsize=(13,7.5),facecolor=C_BG)
fig.canvas.manager.set_window_title(
    f"Multi-Seizure Compilation — {PATIENT}")
gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1.6,1],
                      hspace=0.48,wspace=0.35)
ax_eeg=fig.add_subplot(gs[0,:]); ax_prob=fig.add_subplot(gs[1,0:2])
ax_stat=fig.add_subplot(gs[1,2])
for ax in [ax_eeg,ax_prob,ax_stat]: ax.set_facecolor(C_BG)

ax_eeg.set_title(f"EEG — Channel {DISPLAY_CHANNEL}  ({PATIENT})",
                  fontsize=12,fontweight="bold")
ax_eeg.set_xlabel("Time into segment (seconds)",fontsize=9)
ax_eeg.set_ylabel("Amplitude (uV)",fontsize=9)
ax_eeg.set_ylim(-150,150); ax_eeg.grid(True,alpha=0.4)
eeg_line,=ax_eeg.plot([],[],color="#2166ac",lw=0.7,zorder=3)
vl_alert=ax_eeg.axvline(999999,color=C_ALERT,  lw=2.5,ls="--",zorder=5)
vl_sz_on=ax_eeg.axvline(999999,color=C_SEIZURE,lw=2.5,ls="-", zorder=5)
vl_sz_end=ax_eeg.axvline(999999,color=C_END,   lw=2.5,ls="--",zorder=5)
lbl_alert=ax_eeg.text(0,128,"ALERT",  color=C_ALERT,
                       fontsize=8,fontweight="bold",visible=False,zorder=6)
lbl_sz_on=ax_eeg.text(0,128,"SEIZURE",color=C_SEIZURE,
                       fontsize=8,fontweight="bold",visible=False,zorder=6)
lbl_sz_end=ax_eeg.text(0,128,"END",   color=C_END,
                        fontsize=8,fontweight="bold",visible=False,zorder=6)
ax_eeg.legend(handles=[
    Line2D([0],[0],color=C_ALERT,  lw=2,ls="--",label="Alert fired"),
    Line2D([0],[0],color=C_SEIZURE,lw=2,ls="-", label="Seizure onset"),
    Line2D([0],[0],color=C_END,    lw=2,ls="--",label="Seizure end"),
],fontsize=8,loc="upper right",facecolor=C_BG,edgecolor="#cccccc")

ax_prob.set_title("Preictal Probability",fontsize=12,fontweight="bold")
ax_prob.set_xlabel("Time into segment (seconds)",fontsize=9)
ax_prob.set_ylabel("Probability",fontsize=9)
ax_prob.set_ylim(-0.05,1.1); ax_prob.grid(True,alpha=0.4)
ax_prob.axhline(THRESHOLD,color=C_THRESH,lw=2,ls="--",
                label=f"Alert threshold ({THRESHOLD})",zorder=3)
ax_prob.legend(fontsize=8,loc="upper left",facecolor=C_BG,edgecolor="#cccccc")
prob_line,=ax_prob.plot([],[],color=C_NORMAL,lw=2.5,zorder=4)
pvl_alert=ax_prob.axvline(999999,color=C_ALERT,  lw=2,ls="--",alpha=0)
pvl_sz_on=ax_prob.axvline(999999,color=C_SEIZURE,lw=2,ls="-", alpha=0)
pvl_sz_end=ax_prob.axvline(999999,color=C_END,   lw=2,ls="--",alpha=0)

ax_stat.axis("off"); ax_stat.set_title("Status",fontsize=10,fontweight="bold")
seizure_counter=ax_stat.text(0.5,0.90,"",ha="center",va="center",
                              fontsize=11,color="#888888",
                              transform=ax_stat.transAxes)
status_txt=ax_stat.text(0.5,0.68,"MONITORING",ha="center",va="center",
                         fontsize=15,fontweight="bold",color=C_NORMAL,
                         transform=ax_stat.transAxes)
sub_txt=ax_stat.text(0.5,0.50,"",ha="center",va="center",
                      fontsize=11,color=C_TEXT,transform=ax_stat.transAxes)
detail_txt=ax_stat.text(0.5,0.30,"",ha="center",va="center",
                         fontsize=9,color=C_ALERT,transform=ax_stat.transAxes)
metric_txt=ax_stat.text(0.5,0.08,"",ha="center",va="center",
                         fontsize=8.5,color=C_TEXT,transform=ax_stat.transAxes)
for sp in ax_stat.spines.values():
    sp.set_visible(True); sp.set_edgecolor("#cccccc"); sp.set_linewidth(1.5)

# Time-skip overlay text (shown between seizures)
skip_txt=ax_eeg.text(0.5,0.5,"",ha="center",va="center",
                      fontsize=18,fontweight="bold",color=C_TEXT,
                      transform=ax_eeg.transAxes,
                      bbox=dict(boxstyle="round,pad=0.5",
                                facecolor="#f0f0f0",
                                edgecolor="#cccccc",alpha=0))

fig.suptitle(f"Seizure Prediction  |  {PATIENT}  |  "
              f"Multi-Seizure Compilation  |  v6 Neural Network",
              fontsize=10,color=C_TEXT,y=0.99)
plt.tight_layout(rect=[0,0,1,0.97])


# STATE
def make_seg_state(seg_idx):
    seg=segments[seg_idx]
    total_=seg["norm_seg"].shape[1]
    ds_=0   # segments start at 0
    return {
        "seg_idx":    seg_idx,
        "seg":        seg,
        "raw_seg":    seg["raw_seg"],
        "norm_seg":   seg["norm_seg"],
        "total":      total_,
        "demo_end":   total_,
        "cur":        ds_,
        "paused":     False,
        "speed":      SPEED,
        "eeg_buf":    deque(maxlen=DISPLAY_SAMP),
        "t_buf":      deque(maxlen=DISPLAY_SAMP),
        "probs":      [],
        "times":      [],
        "alerted":    False,
        "alert_fired_t": None,
        "sz_shown":   False,
        "end_shown":  False,
        "alert_drawn":False,
        "sz_drawn":   False,
        "sz_end_drawn":False,
        "last_win":   -STEP_SAMPLES,
        "skip_mode":  False,
        "skip_frames":0,
    }

state=make_seg_state(0)

def load_segment(idx):
    """Switch to a new seizure segment."""
    global state
    state=make_seg_state(idx)
    # Reset bars
    for vl in [vl_alert,vl_sz_on,vl_sz_end]: vl.set_xdata([999999])
    for pvl in [pvl_alert,pvl_sz_on,pvl_sz_end]:
        pvl.set_xdata([999999]); pvl.set_alpha(0)
    for lbl in [lbl_alert,lbl_sz_on,lbl_sz_end]: lbl.set_visible(False)
    prob_line.set_data([],[]); eeg_line.set_data([],[])
    ax_prob.set_xlim(0,10)

def on_key(e):
    if e.key==" ":       state["paused"]=not state["paused"]
    elif e.key=="q":     plt.close()
    elif e.key in["+","=","right"]:
        state["speed"]=min(state["speed"]*1.5,120)
    elif e.key in["-","left"]:
        state["speed"]=max(state["speed"]/1.5,0.5)

fig.canvas.mpl_connect("key_press_event",on_key)

SKIP_FRAMES=45   # ~1.5 sec of "TIME SKIP" screen at 30fps
FRAME_MS=33


def update(frame):
    if state["paused"]: return
    seg=state["seg"]
    sz_num=seg["sz_num"]

    # TIME SKIP mode — show transition screen between seizures
    if state["skip_mode"]:
        state["skip_frames"]+=1
        if state["skip_frames"]>=SKIP_FRAMES:
            # Move to next segment
            next_idx=state["seg_idx"]+1
            if next_idx>=len(segments):
                next_idx=0   # loop back to start
            skip_txt.set_text(""); skip_txt.get_bbox_patch().set_alpha(0)
            ax_eeg.set_facecolor(C_BG)
            load_segment(next_idx)
        return

    # Normal playback
    adv=max(1,int(state["speed"]*(FRAME_MS/1000)*TARGET_SFREQ))
    ns=state["cur"]+adv

    if ns>=state["total"]:
        # Segment done — show TIME SKIP screen
        state["skip_mode"]=True
        state["skip_frames"]=0
        next_num=(state["seg_idx"]+1)%len(segments)+1
        skip_txt.set_text(
            f"--- TIME SKIP ---\n"
            f"Seizure {sz_num} of {len(segments)} complete\n"
            f"Loading Seizure {next_num}...")
        skip_txt.get_bbox_patch().set_alpha(0.9)
        ax_eeg.set_facecolor("#f8f8f8")
        return

    state["cur"]=ns
    t=ns/TARGET_SFREQ

    # Feed EEG (display channel only)
    e0=min(ns+adv,state["total"])
    rn=state["raw_seg"][seg["dci"],ns:e0].copy()
    if np.max(np.abs(rn))<0.01: rn*=1e6
    state["eeg_buf"].extend(rn.tolist())
    state["t_buf"].extend((np.arange(ns,e0)/TARGET_SFREQ).tolist())

    # Run model
    while (state["cur"]-state["last_win"])>=STEP_SAMPLES:
        ws=state["last_win"]+STEP_SAMPLES
        we=ws+WINDOW_SAMPLES; state["last_win"]=ws
        if we>state["total"]: break
        prob=run_model(state["norm_seg"][:,ws:we],
                       seg["nch"],seg["bmean"],seg["bstd"],model,scaler)
        wt=we/TARGET_SFREQ
        state["probs"].append(prob); state["times"].append(wt)
        if prob>=THRESHOLD and not state["alerted"]:
            state["alerted"]=True; state["alert_fired_t"]=t
            print(f"  [SZ {sz_num}] ALERT horizon="
                  f"{seg['horizon']:.0f}s ({seg['horizon']/60:.1f}min)")
        if t>=seg["onset_sec"] and not state["sz_shown"]:
            state["sz_shown"]=True
        if t>=seg["end_sec"] and not state["end_shown"]:
            state["end_shown"]=True

    # EEG plot
    ta=np.array(state["t_buf"]); ea=np.array(state["eeg_buf"])
    if len(ta)>1:
        eeg_line.set_data(ta,ea)
        ax_eeg.set_xlim(max(0,ta[-1]-DISPLAY_SEC),ta[-1]+5)

        if state["alerted"] and not state["alert_drawn"]:
            at=state["alert_fired_t"]
            vl_alert.set_xdata([at]); pvl_alert.set_xdata([at])
            pvl_alert.set_alpha(0.9)
            lbl_alert.set_position((at+1,128)); lbl_alert.set_visible(True)
            state["alert_drawn"]=True

        if t>=seg["onset_sec"] and not state["sz_drawn"]:
            vl_sz_on.set_xdata([seg["onset_sec"]])
            pvl_sz_on.set_xdata([seg["onset_sec"]]); pvl_sz_on.set_alpha(0.9)
            lbl_sz_on.set_position((seg["onset_sec"]+1,128))
            lbl_sz_on.set_visible(True); state["sz_drawn"]=True

        if t>=seg["end_sec"] and not state["sz_end_drawn"]:
            vl_sz_end.set_xdata([seg["end_sec"]])
            pvl_sz_end.set_xdata([seg["end_sec"]]); pvl_sz_end.set_alpha(0.9)
            lbl_sz_end.set_position((seg["end_sec"]+1,128))
            lbl_sz_end.set_visible(True); state["sz_end_drawn"]=True

    # Probability
    if state["probs"]:
        ph=np.array(state["probs"]); th=np.array(state["times"])
        prob_line.set_data(th,ph); cp=ph[-1]
        prob_line.set_color(C_ALERT if cp>=THRESHOLD else
                            C_CAUTION if cp>=0.4 else C_NORMAL)
        ax_prob.set_xlim(max(0,th[-1]-DISPLAY_SEC),th[-1]+5)

    # Status panel
    cp=state["probs"][-1] if state["probs"] else 0.0
    pct=int(cp*100); spd=f"{state['speed']:.0f}x"
    col_m=(C_ALERT if cp>=THRESHOLD else C_CAUTION if cp>=0.4 else C_NORMAL)
    h=seg["horizon"]

    seizure_counter.set_text(
        f"Seizure {sz_num} of {len(segments)}")

    if t>=seg["end_sec"]:
        status_txt.set_text("MONITORING"); status_txt.set_color(C_NORMAL)
        sub_txt.set_text("Seizure ended\nBack to monitoring")
        sub_txt.set_color(C_NORMAL); detail_txt.set_text("")
        ax_stat.set_facecolor(C_BG)
    elif t>=seg["onset_sec"]:
        status_txt.set_text("SEIZURE"); status_txt.set_color(C_SEIZURE)
        sub_txt.set_text("Seizure in progress"); sub_txt.set_color(C_SEIZURE)
        detail_txt.set_text(f"Predicted {h/60:.1f} min early" if h>0 else "")
        detail_txt.set_color(C_SEIZURE); ax_stat.set_facecolor("#f8f0ff")
    elif state["alerted"]:
        el=max(0,t-state["alert_fired_t"])
        am=int(el//60); as2=int(el%60)
        status_txt.set_text(">>> ALERT <<<"); status_txt.set_color(C_ALERT)
        sub_txt.set_text(f"Alert active: {am}m {as2:02d}s")
        sub_txt.set_color(C_ALERT)
        detail_txt.set_text(f"Model crossed {int(THRESHOLD*100)}% threshold\n")
        detail_txt.set_color(C_ALERT); ax_stat.set_facecolor("#fff5f5")
    else:
        status_txt.set_text("MONITORING"); status_txt.set_color(C_NORMAL)
        sub_txt.set_text(""); detail_txt.set_text("")
        ax_stat.set_facecolor(C_BG)

    metric_txt.set_text(f"Certainty: {pct}%   Threshold: {int(THRESHOLD*100)}%\n"
                         f"AUC: {PATIENT_AUC} ({PATIENT})   Speed: {spd}")
    metric_txt.set_color(col_m)
    fig.canvas.draw_idle()


# RUN
print(f"  Controls: SPACE=pause  RIGHT/+=faster  LEFT/-=slower  Q=quit")
print(f"  {len(segments)} seizures will play in sequence with TIME SKIP between\n")

ani=FuncAnimation(fig,update,interval=FRAME_MS,cache_frame_data=False,blit=False)
plt.show()