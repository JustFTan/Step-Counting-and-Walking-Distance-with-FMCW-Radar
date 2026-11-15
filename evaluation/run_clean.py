import json,time,numpy as np,pickle,sys
from common import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
T=load_trials(); FS=1000/55.0; F=['xvel','yvel','zvel','vmag']
# cleaned primary track, resampled to uniform 55 ms grid
for t in T:
    d,_=primary_track(t['df']); ts=(d.timestamp.values-d.timestamp.values[0])/1000.0
    keep=np.r_[True,np.diff(ts)>1e-3]; d=d[keep]; ts=ts[keep]; tu=np.arange(0,ts[-1],1/FS)
    c=pd.DataFrame({k:np.interp(tu,ts,d[k].interpolate().bfill().values) for k in ['xvel','yvel','zvel']}); c['vmag']=np.sqrt((c**2).sum(1)); t['df']=c
def ev(scheme,height):
    feats=F+(['height'] if height else [])
    Xs,ys,g,s=[],[],[],[]
    for t in T:
        if len(t['df'])<W: continue
        X,y=windows(t,feats); Xs.append(X);ys.append(y);g+=[t['id']]*len(X);s+=[t['subj']]*len(X)
    X=np.vstack(Xs);y=np.vstack(ys);g=np.array(g);s=np.array(s)
    folds=list(GroupKFold(5).split(X,y,g)) if scheme=="trial" else [(np.where(s!=u)[0],np.where(s==u)[0]) for u in np.unique(s)]
    out=[]
    for tr,te in folds:
        m=ExtraTreesRegressor(100,n_jobs=2,random_state=0).fit(X[tr],y[tr])
        for t in T:
            if t['id'] in set(g[te]):
                Xt,_=windows(t,feats,stride=W); p=m.predict(Xt).sum(0)*len(t['df'])/(len(Xt)*W)
                out.append(dict(id=t['id'],subj=t['subj'],scen=t['scen'],N=t['N'],D=t['D'],pN=p[0],pD=p[1]))
    o=pd.DataFrame(out); o.to_csv(f"clean_{scheme}_{'h' if height else 'noh'}.csv",index=False)
    print(scheme,height,"n",len(o),"stepMAE %.2f bias %+.2f | distMAE %.2f bias %+.2f"%((o.pN-o.N).abs().mean(),(o.pN-o.N).mean(),(o.pD-o.D).abs().mean(),(o.pD-o.D).mean()),flush=True)
for a in sys.argv[1:]:
    sc,h=a.split(":"); ev(sc,h=="h")
