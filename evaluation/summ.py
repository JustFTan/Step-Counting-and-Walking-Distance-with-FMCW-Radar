import json,glob,numpy as np,pandas as pd
B=pd.read_csv("baselines.csv"); pd.set_option("display.width",250)
rows=[]
def add(name,scheme,df,pn,pd_):
    eN=df[pn]-df.N; eD=df[pd_]-df.D
    r=dict(method=name,scheme=scheme,n=len(df),stepMAE=eN.abs().mean(),stepBias=eN.mean(),stepMAPE=100*(eN.abs()/df.N).mean(),distMAE=eD.abs().mean(),distBias=eD.mean(),distMAPE=100*(eD.abs()/df.D).mean())
    for sc in sorted(df.scen.unique()):
        m=df.scen==sc; r["S_"+sc]=eN[m].abs().mean(); r["D_"+sc]=eD[m].abs().mean()
    rows.append(r)
allp={}
for f in sorted(glob.glob("res_*.json")):
    tag=f[4:-5]; R=json.load(open(f))
    for nm,r in R.items():
        if r['trials']:
            df=pd.DataFrame(r['trials']); add(nm,tag,df,'pN','pD'); add(nm+"(cov)",tag,df,'pNc','pDc'); allp[(nm,tag)]=df
add("Trajectory path + step-length(k*H)","loso-param",B,'steplen_N','path')
add("Calibrated path / duration-rate","loso-param",B,'dur_N','path_cal')
add("Cadence FFT / vel-integration","none",B,'fft','vint')
add("Cadence peaks / raw path","none",B,'peaks','path_raw')
add("Mean-rate x frames","loso-param",B,'meanrate_N','meanrate_D')
S=pd.DataFrame(rows); S.to_csv("summary_trials.csv",index=False)
print(S[['method','scheme','n','stepMAE','stepBias','stepMAPE','distMAE','distBias','distMAPE']].round(2).to_string())
print(S[['method','scheme']+[c for c in S.columns if c[:2] in('S_','D_')]].round(2).to_string())
# per subject for LOSO ET
for k in [("ET","loso_h"),("ET","loso_noh")]:
    d=allp[k]; d['eN']=(d.pN-d.N).abs(); d['eD']=(d.pD-d.D).abs(); print(k,d.groupby('subj')[['eN','eD']].mean().round(2).to_dict())
B['eN']=(B.steplen_N-B.N).abs(); B['eD']=(B.path-B.D).abs(); print("traj",B.groupby('subj')[['eN','eD']].mean().round(2).to_dict())
