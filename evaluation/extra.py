import sys,json,numpy as np
import common
from common import *
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
T=load_trials(); FS=1000/55.0; F=['xvel','yvel','zvel','vmag']
for t in T:
    d,_=primary_track(t['df']); ts=(d.timestamp.values-d.timestamp.values[0])/1000.0
    keep=np.r_[True,np.diff(ts)>1e-3]; d=d[keep]; ts=ts[keep]; tu=np.arange(0,ts[-1],1/FS)
    c=pd.DataFrame({k:np.interp(tu,ts,d[k].interpolate().bfill().values) for k in ['xvel','yvel','zvel']}); c['vmag']=np.sqrt((c**2).sum(1)); t['df']=c
MODELS={"ET":lambda:ExtraTreesRegressor(100,n_jobs=2,random_state=0),"RF":lambda:RandomForestRegressor(60,max_features=0.33,n_jobs=2,random_state=0),
 "KNN":lambda:KNeighborsRegressor(3,weights='distance',metric='manhattan'),"HGB":lambda:MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0)),
 "Ridge":lambda:make_pipeline(StandardScaler(),Ridge(10.0)),"Mean":lambda:DummyRegressor()}
def loso(name,W):
    common.W=W
    Xs,ys,s=[],[],[]
    for t in T:
        if len(t['df'])<W: continue
        X,y=windows(t,F); Xs.append(X);ys.append(y);s+=[t['subj']]*len(X)
    X=np.vstack(Xs);y=np.vstack(ys);s=np.array(s); out=[]
    for u in np.unique(s):
        m=MODELS[name]().fit(X[s!=u],y[s!=u])
        for t in T:
            if t['subj']==u and len(t['df'])>=W:
                Xt,_=windows(t,F,stride=W); p=m.predict(Xt).sum(0)*len(t['df'])/(len(Xt)*W)
                out.append(dict(id=t['id'],subj=u,scen=t['scen'],N=t['N'],D=t['D'],pN=p[0],pD=p[1]))
    o=pd.DataFrame(out); o.to_csv(f"cleanx_{name}_W{W}.csv",index=False)
    print(name,W,len(o),"step %.2f dist %.2f"%((o.pN-o.N).abs().mean(),(o.pD-o.D).abs().mean()),flush=True)
for a in sys.argv[1:]:
    n,w=a.split(":"); loso(n,int(w))
