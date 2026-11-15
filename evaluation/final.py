import json,time,pickle,os,numpy as np,pandas as pd
from scipy.stats import wilcoxon
rng=np.random.RandomState(1); B=pd.read_csv("baselines.csv")
M={}
def reg(name,df,pn,pd_): d=df.sort_values('id'); M[name]=(d[pn].values-d.N.values,d[pd_].values-d.D.values,d)
for tag,nm in [("loso_h","Original pipeline (raw rows, +height), LOSO"),("loso_noh","Original pipeline (raw rows, no height), LOSO"),("trial_h","Original pipeline (raw rows, +height), trial-grouped 5-fold")]:
    reg(nm,pd.DataFrame(json.load(open(f"res_{tag}.json"))["ET"]["trials"]),'pN','pD')
reg("Mean-rate predictor, LOSO",pd.DataFrame(json.load(open("res_loso_h.json"))["Mean"]["trials"]),'pN','pD')
for f,nm in [("clean_loso_noh","Cleaned track ET (no height), LOSO"),("clean_loso_h","Cleaned track ET (+height), LOSO"),("clean_trial_noh","Cleaned track ET (no height), trial-grouped"),("clean_trial_h","Cleaned track ET (+height), trial-grouped")]:
    reg(nm,pd.read_csv(f+".csv"),'pN','pD')
reg("Trajectory baseline (path length; steps=path/(k*H)), LOSO-k",B,'steplen_N','path')
reg("Trajectory baseline, calibrated path (LOSO gain)",B,'steplen_N','path_cal')
reg("Cadence FFT (steps) / velocity integration (dist)",B,'fft','vint')
def ci(a):
    bs=[np.abs(a[rng.randint(0,len(a),len(a))]).mean() for _ in range(5000)]; return np.percentile(bs,[2.5,97.5])
rows=[]
for k,(eN,eD,d) in M.items():
    cN,cD=ci(eN),ci(eD)
    r=dict(method=k,stepMAE=np.abs(eN).mean(),stepCI=f"[{cN[0]:.2f}, {cN[1]:.2f}]",stepBias=eN.mean(),stepLoA=f"[{eN.mean()-1.96*eN.std(ddof=1):.2f}, {eN.mean()+1.96*eN.std(ddof=1):.2f}]",stepRelErr=100*(np.abs(eN)/d.N.values).mean(),
           distMAE=np.abs(eD).mean(),distCI=f"[{cD[0]:.2f}, {cD[1]:.2f}]",distBias=eD.mean(),distLoA=f"[{eD.mean()-1.96*eD.std(ddof=1):.2f}, {eD.mean()+1.96*eD.std(ddof=1):.2f}]",distRelErr=100*(np.abs(eD)/d.D.values).mean())
    for sc in sorted(d.scen.unique()):
        m=(d.scen==sc).values; r["step|"+sc]=np.abs(eN[m]).mean(); r["dist|"+sc]=np.abs(eD[m]).mean()
    for u in sorted(d.subj.unique()):
        m=(d.subj==u).values; r["step|"+u]=np.abs(eN[m]).mean(); r["dist|"+u]=np.abs(eD[m]).mean()
    rows.append(r)
S=pd.DataFrame(rows); S.to_csv("final_trial_level.csv",index=False); pd.set_option("display.width",300); pd.set_option("display.max_colwidth",60)
print(S.iloc[:,:11].round(2).to_string()); print(S.iloc[:,[0]+list(range(11,S.shape[1]))].round(2).to_string())
a="Cleaned track ET (no height), LOSO"
for b in ["Trajectory baseline (path length; steps=path/(k*H)), LOSO-k","Original pipeline (raw rows, +height), LOSO","Mean-rate predictor, LOSO","Cleaned track ET (+height), LOSO","Trajectory baseline, calibrated path (LOSO gain)"]:
    print("Wilcoxon |err|",a,"vs",b,"steps p=%.3g dist p=%.3g"%(wilcoxon(np.abs(M[a][0]),np.abs(M[b][0])).pvalue,wilcoxon(np.abs(M[a][1]),np.abs(M[b][1])).pvalue))
W=pd.DataFrame([dict(scheme=f[4:-5],model=k,**{m:v[m] for m in v if m.startswith('win')}) for f in sorted(__import__('glob').glob("res_*.json")) for k,v in json.load(open(f)).items()])
W.to_csv("final_window_level.csv",index=False); print(W.round(3).to_string())
