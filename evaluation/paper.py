import json,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from common import *
rng=np.random.RandomState(1); B=pd.read_csv("baselines.csv").sort_values('id')
def J(tag,m,c=True):
    d=pd.DataFrame(json.load(open(f"res_{tag}.json"))[m]["trials"]).sort_values('id')
    return d.assign(pN=d.pNc,pD=d.pDc) if c else d
def C(f): return pd.read_csv(f).sort_values('id')
M=[("O-rand","Original (raw rows, +height) ET",None,None),
 ("A","Original pipeline: ET, raw rows, +height","LOSO",J("loso_h","ET")),
 ("A2","Original pipeline: ET, raw rows, no height","LOSO",J("loso_noh","ET")),
 ("A3","Mean-rate predictor, raw rows","LOSO",J("loso_h","Mean")),
 ("A4","Original pipeline: ET, raw rows, +height","Trial-grouped",J("trial_h","ET")),
 ("B1","Cleaned track: mean-rate predictor","LOSO",C("cleanx_Mean_W23.csv")),
 ("B2","Cleaned track: Ridge","LOSO",C("cleanx_Ridge_W23.csv")),
 ("B3","Cleaned track: KNN","LOSO",C("cleanx_KNN_W23.csv")),
 ("B4","Cleaned track: HGB","LOSO",C("cleanx_HGB_W23.csv")),
 ("B5","Cleaned track: RF","LOSO",C("cleanx_RF_W23.csv")),
 ("B6","Cleaned track: ET","LOSO",C("clean_loso_noh.csv")),
 ("B7","Cleaned track: ET +height","LOSO",C("clean_loso_h.csv")),
 ("B8","Cleaned track: ET","Trial-grouped",C("clean_trial_noh.csv")),
 ("B9","Cleaned track: ET, W=12","LOSO",C("cleanx_ET_W12.csv")),
 ("B10","Cleaned track: ET, W=35","LOSO",C("cleanx_ET_W35.csv")),
 ("T1","Trajectory baseline (path; steps=path/(kH))","LOSO (k)",B.assign(pN=B.steplen_N,pD=B.path)),
 ("T2","Trajectory baseline, calibrated gain","LOSO (k,a)",B.assign(pN=B.steplen_N,pD=B.path_cal)),
 ("T3","Cadence FFT / velocity integration","none",B.assign(pN=B.fft,pD=B.vint))]
def ci(a):
    bs=[np.abs(a[rng.randint(0,len(a),len(a))]).mean() for _ in range(5000)]; return np.percentile(bs,[2.5,97.5])
rows=[];E={}
for k,nm,sch,d in M:
    if d is None: continue
    eN=(d.pN-d.N).values;eD=(d.pD-d.D).values;E[k]=(eN,eD,d);cN,cD=ci(eN),ci(eD)
    r=dict(key=k,method=nm,protocol=sch,sMAE=np.abs(eN).mean(),sLo=cN[0],sHi=cN[1],sBias=eN.mean(),sSD=eN.std(ddof=1),sRel=100*(np.abs(eN)/d.N.values).mean(),dMAE=np.abs(eD).mean(),dLo=cD[0],dHi=cD[1],dBias=eD.mean(),dSD=eD.std(ddof=1),dRel=100*(np.abs(eD)/d.D.values).mean())
    for g in ['scen','subj']:
        for v in sorted(d[g].unique()):
            m=(d[g]==v).values; r["s|"+v]=np.abs(eN[m]).mean(); r["d|"+v]=np.abs(eD[m]).mean()
    rows.append(r)
S=pd.DataFrame(rows); S.to_csv("paper_table.csv",index=False); pd.set_option("display.width",320)
print(S.iloc[:,[0,2]+list(range(3,15))].round(2).to_string()); print(S.iloc[:,[0]+list(range(15,S.shape[1]))].round(2).to_string())
for a,b in [("B6","B1"),("B6","T1"),("B6","T2"),("B6","A"),("B6","B7"),("B6","B5"),("B6","B3"),("B6","B4"),("B6","B2"),("A","A3"),("B6","B9"),("B6","B10")]:
    print(a,"vs",b,"steps p=%.2g dist p=%.2g"%(wilcoxon(np.abs(E[a][0]),np.abs(E[b][0])).pvalue,wilcoxon(np.abs(E[a][1]),np.abs(E[b][1])).pvalue))
# dataset stats
T=load_trials(); tot=sum(len(t['df']) for t in T); fr=[primary_track(t['df'])[1] for t in T]
print("rows",tot,"primary-track share mean %.3f; per subj"%np.mean(fr),{u:round(np.mean([f for f,t in zip(fr,T) if t['subj']==u]),2) for u in ['S1','S2','S3']})
print(B.groupby('scen')[['dur','frames']].agg(['mean','std']).round(1)); print(B.groupby('subj').size().to_dict(), B.groupby('scen').size().to_dict(), "cadence true steps/s",(B.N/B.dur).groupby(B.subj).mean().round(2).to_dict(),"step len",(B.D/B.N).groupby(B.scen).mean().round(2).to_dict())
# ---------- figures
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"pdf.fonttype":42})
BL="#2a78d6";INK="#0b0b0b";MUT="#52514e"
import os; os.makedirs("Fig",exist_ok=True)
Wn=pd.read_csv("final_window_level.csv").set_index(['scheme','model'])
fig,ax=plt.subplots(figsize=(3.5,2.3)); labs=["Random\nsplit","Shuffled\nlabels","Trial-\ngrouped","LOSO"];sch=["random_h","shuf_random","trial_h","loso_h"]
x=np.arange(4);w=0.36
for i,(m,kw) in enumerate([("ET",dict(color=BL)),("KNN",dict(facecolor="white",edgecolor=BL,hatch="////"))]):
    v=[Wn.loc[(s,m),'win_r2_steps'] for s in sch]; b=ax.bar(x+(i-.5)*(w+0.03),v,w,label=m,linewidth=0.8,**kw)
    for xi,vi in zip(x+(i-.5)*(w+0.03),v): ax.text(xi,vi+(0.04 if vi>=0 else -0.04),f"{vi:.2f}",ha="center",va="bottom" if vi>=0 else "top",fontsize=6.5,color=INK)
ax.axhline(0,color=MUT,lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Per-window $R^2$ (steps)"); ax.set_ylim(-1.75,1.2); ax.legend(frameon=False,loc="lower left",ncol=2)
fig.tight_layout(); fig.savefig("Fig/leakage_r2.pdf"); fig.savefig("Fig/leakage_r2.png",dpi=300); plt.close()
# ghost-track example
t=[t for t in T if t['subj']=='S3' and t['scen'].startswith('8')][0]; d=t['df']; p,_=primary_track(d); ts=(d.timestamp-d.timestamp.iloc[0])/1000
fig,ax=plt.subplots(figsize=(3.5,2.1)); ax.plot(ts,d.xpos,'.',ms=3,color=MUT,alpha=.55,label="All logged tracker rows"); ax.plot((p.timestamp-d.timestamp.iloc[0])/1000,p.xpos,'-',lw=1.6,color=BL,label="Associated primary track")
ax.set_xlabel("Time (s)");ax.set_ylabel("Tracker $x$ position (m)");ax.legend(frameon=False,fontsize=7,loc="best"); fig.tight_layout(); fig.savefig("Fig/ghost_track.pdf"); fig.savefig("Fig/ghost_track.png",dpi=300); plt.close()
print("ghost example", t['file'], len(d), len(p))
# forest plot
order=["A","A3","A4","B1","B6","B8","T1","T2"]; names=["Original pipeline (LOSO)","Mean-rate, raw rows (LOSO)","Original pipeline (trial-grouped)","Mean-rate, cleaned track (LOSO)","ET, cleaned track (LOSO)","ET, cleaned track (trial-grouped)","Trajectory baseline (LOSO)","Trajectory, calibrated (LOSO)"]
Sx=S.set_index('key'); fig,axs=plt.subplots(1,2,figsize=(7.16,2.3),sharey=True)
for ax,(c,lab) in zip(axs,[("s","Step-count MAE per trial (steps)"),("d","Distance MAE per trial (m)")]):
    for i,k in enumerate(order):
        r=Sx.loc[k]; y=len(order)-1-i; ax.plot([r[c+'Lo'],r[c+'Hi']],[y,y],color=BL,lw=1.4); ax.plot(r[c+'MAE'],y,'o',ms=5,color=BL,mec="white",mew=0.8); ax.text(r[c+'Hi']+0.08*(1 if c=='s' else .5),y,f"{r[c+'MAE']:.2f}",va="center",fontsize=7,color=INK)
    ax.set_xlabel(lab); ax.set_xlim(0,None); ax.grid(axis='x',color="#e6e5e0",lw=0.5); ax.set_axisbelow(True)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(names[::-1]); fig.tight_layout(); fig.savefig("Fig/trial_mae.pdf"); fig.savefig("Fig/trial_mae.png",dpi=300); plt.close()
# Bland-Altman
eN,eD,d=E["B6"]; fig,axs=plt.subplots(1,2,figsize=(7.16,2.2))
for ax,(e,tr,pr,u) in zip(axs,[(eN,d.N.values,d.pN.values,"steps"),(eD,d.D.values,d.pD.values,"m")]):
    ax.plot((tr+pr)/2,e,'o',ms=3.5,color=BL,mec="white",mew=0.4,alpha=.9); b=e.mean();s=e.std(ddof=1)
    for v,ls,tx in [(b,'-',f"bias {b:+.2f}"),(b+1.96*s,'--',f"+1.96 SD {b+1.96*s:+.2f}"),(b-1.96*s,'--',f"$-$1.96 SD {b-1.96*s:+.2f}")]:
        ax.axhline(v,color=MUT,lw=0.7,ls=ls); ax.text(ax.get_xlim()[1],v,tx,ha="right",va="bottom",fontsize=6.5,color=INK)
    ax.set_xlabel(f"Mean of estimate and ground truth ({u})"); ax.set_ylabel(f"Estimate $-$ truth ({u})")
fig.tight_layout(); fig.savefig("Fig/bland_altman.pdf"); fig.savefig("Fig/bland_altman.png",dpi=300); plt.close()
