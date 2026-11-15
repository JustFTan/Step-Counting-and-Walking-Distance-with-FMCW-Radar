import os,re,glob,json,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder IWR6843AOP
W=23
def load_trials(base="Dataset_All"):
    T=[]
    for f in sorted(glob.glob(f"{ROOT}/{base}/*/*/*.csv")):
        subj,fo=f.split("/")[-3:-1]; m=re.match(r"([\d.]+)m_(\d+)_(\d+)steps",fo)
        d=pd.read_csv(f)
        d['vmag']=np.sqrt(d.xvel**2+d.yvel**2+d.zvel**2)
        T.append(dict(id=len(T),subj=subj,scen=f"{m.group(1)}m/{m.group(3)}st",D=float(m.group(1)),H=float(m.group(2)),N=float(m.group(3)),df=d,file=f))
    return T
def windows(t,feats,stride=1):
    d=t['df'].copy(); d['height']=t['H']; n=len(d)
    X=[d[feats].iloc[i:i+W].values.ravel() for i in range(0,n-W+1,stride)]
    X=np.array(X).reshape(len(X),W*len(feats))
    y=np.tile([t['N']/n*W,t['D']/n*W],(len(X),1))
    return X,y
def primary_track(d,gate=0.4):
    """greedy nearest-neighbour chaining; returns rows of the longest chain"""
    P=d[['xpos','ypos']].values; chains=[]
    for i,p in enumerate(P):
        if np.isnan(p).any(): continue
        best=None;bd=gate
        for c in chains:
            if i-c[-1]>12: continue
            dist=np.linalg.norm(p-P[c[-1]])
            if dist<bd: bd=dist;best=c
        if best is None: chains.append([i])
        else: best.append(i)
    c=max(chains,key=len)
    return d.iloc[c].dropna(subset=['timestamp','xpos','ypos','zpos']).reset_index(drop=True), len(c)/len(d)
