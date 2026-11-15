import json,numpy as np
from common import *
from scipy.signal import butter,filtfilt,find_peaks
T=load_trials(); FS=1000/55.0
rows=[]
for t in T:
    d,frac=primary_track(t['df'])
    ts=(d.timestamp.values-d.timestamp.values[0])/1000.0
    keep=np.r_[True,np.diff(ts)>1e-3]; d=d[keep]; ts=ts[keep]
    tu=np.arange(0,ts[-1],1/FS)
    x,y,z=[np.interp(tu,ts,d[c].values) for c in ('xpos','ypos','zpos')]
    k=5; sm=lambda v:np.convolve(np.pad(v,(k//2,k//2),mode='edge'),np.ones(k)/k,mode='valid')
    xs,ys=sm(x),sm(y)
    path=float(np.hypot(np.diff(xs),np.diff(ys)).sum()); path_raw=float(np.hypot(np.diff(x),np.diff(y)).sum())
    vx,vy=[np.interp(tu,ts,d[c].fillna(0).values) for c in ('xvel','yvel')]
    vint=float(np.hypot(vx,vy).sum()/FS)
    # cadence from vertical oscillation of tracked centroid
    pk=np.nan;fN=np.nan
    if len(tu)>30:
        b,a=butter(2,[1.0/(FS/2),3.0/(FS/2)],btype='band'); zf=filtfilt(b,a,z-z.mean())
        p,_=find_peaks(zf,distance=int(0.33*FS)); pk=float(len(p))
        S=np.abs(np.fft.rfft(zf*np.hanning(len(zf)),4096)); fr=np.fft.rfftfreq(4096,1/FS); band=(fr>=1.0)&(fr<=3.0)
        fN=float(fr[band][S[band].argmax()]*tu[-1])
    rows.append(dict(id=t['id'],subj=t['subj'],scen=t['scen'],N=t['N'],D=t['D'],H=t['H'],frames=len(t['df']),dur=float(ts[-1]),frac=frac,path=path,path_raw=path_raw,vint=vint,peaks=pk,fft=fN,net=float(np.hypot(x[-1]-x[0],y[-1]-y[0]))))
R=pd.DataFrame(rows)
# subject-independent parametric baselines (leave-one-subject-out)
for u in R.subj.unique():
    tr=R[R.subj!=u]; te=R.subj==u
    R.loc[te,'meanrate_N']=R.loc[te,'frames']*(tr.N.sum()/tr.frames.sum())
    R.loc[te,'meanrate_D']=R.loc[te,'frames']*(tr.D.sum()/tr.frames.sum())
    R.loc[te,'dur_N']=R.loc[te,'dur']*(tr.N.sum()/tr.dur.sum())
    kk=np.median(tr.D/tr.N/(tr.H/100)); R.loc[te,'steplen_N']=R.loc[te,'path']/(kk*R.loc[te,'H']/100)
    R.loc[te,'k']=kk
    a=np.sum(tr.path*tr.D)/np.sum(tr.path**2); R.loc[te,'path_cal']=a*R.loc[te,'path']; R.loc[te,'a']=a
R.to_csv("baselines.csv",index=False)
pd.set_option("display.width",250)
print(R.groupby(['subj','scen'])[['frac','dur','D','path','path_raw','vint','net','N','peaks','fft','steplen_N','dur_N']].mean().round(2))
def summ(p,t): e=R[p]-R[t]; return f"MAE {e.abs().mean():.2f} bias {e.mean():+.2f} MAPE {100*(e.abs()/R[t]).mean():.1f}%"
for p in ['path','path_cal','path_raw','vint','meanrate_D']: print("DIST",p,summ(p,'D'))
for p in ['peaks','fft','steplen_N','dur_N','meanrate_N']: print("STEP",p,summ(p,'N'))
print(R[['k','a']].drop_duplicates())
