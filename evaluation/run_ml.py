import sys,time,json,numpy as np
from common import *
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GroupKFold
from sklearn.metrics import mean_absolute_error as mae,r2_score as r2
T=load_trials(); rng=np.random.RandomState(0)
def models():
    return {"ET":ExtraTreesRegressor(100,n_jobs=2,random_state=0),
            "RF":RandomForestRegressor(60,max_features=0.33,n_jobs=2,random_state=0),
            "KNN":make_pipeline(KNeighborsRegressor(3,weights='distance',metric='manhattan')),
            "HGB":MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0)),
            "Ridge":make_pipeline(StandardScaler(),Ridge(10.0)),
            "Mean":DummyRegressor()}
def build(feats,labels=None):
    Xs,ys,g,s=[],[],[],[]
    for t in T:
        tt=dict(t)
        if labels is not None: tt['N'],tt['D']=labels[t['id']]
        X,y=windows(tt,feats); Xs.append(X);ys.append(y);g+= [t['id']]*len(X); s+=[t['subj']]*len(X)
    X=SimpleImputer(strategy='mean').fit_transform(np.vstack(Xs))
    return X,np.vstack(ys),np.array(g),np.array(s)
def trial_eval(m,feats,test_trials):
    out=[]
    for t in test_trials:
        X,_=windows(t,feats,stride=W)
        X=np.nan_to_num(X,nan=0.0)
        p=m.predict(X).sum(0); n=len(t['df']); cov=len(X)*W/n
        out.append(dict(id=t['id'],subj=t['subj'],scen=t['scen'],N=t['N'],D=t['D'],pN=float(p[0]),pD=float(p[1]),pNc=float(p[0]/cov),pDc=float(p[1]/cov)))
    return out
def run(tag,feats,scheme,names,labels=None):
    X,y,g,s=build(feats,labels); res=json.load(open(f"res_{tag}.json")) if os.path.exists(f"res_{tag}.json") else {}
    if scheme=="random":
        tr,te=train_test_split(np.arange(len(X)),test_size=0.2,random_state=42); folds=[(tr,te)]
    elif scheme=="trial":
        folds=list(GroupKFold(5).split(X,y,g))
    elif scheme=="loso":
        folds=[(np.where(s!=u)[0],np.where(s==u)[0]) for u in np.unique(s)]
    for nm in names:
        P=np.full_like(y,np.nan); tri=[]; t0=time.time()
        for tr,te in folds:
            m=models()[nm].fit(X[tr],y[tr]); P[te]=m.predict(X[te])
            if scheme!="random":
                ids=set(g[te]); tri+=trial_eval(m,feats,[t for t in T if t['id'] in ids])
        ok=~np.isnan(P[:,0])
        r=dict(win_mae_steps=mae(y[ok,0],P[ok,0]),win_r2_steps=r2(y[ok,0],P[ok,0]),win_mae_dist=mae(y[ok,1],P[ok,1]),win_r2_dist=r2(y[ok,1],P[ok,1]),trials=tri,sec=time.time()-t0)
        if scheme=="loso":
            r['per_subj']={u:dict(r2s=r2(y[s==u,0],P[s==u,0]),maes=mae(y[s==u,0],P[s==u,0]),r2d=r2(y[s==u,1],P[s==u,1])) for u in np.unique(s)}
        res[nm]=r; print(tag,nm,{k:round(v,4) for k,v in r.items() if k.startswith('win')},round(r['sec']),flush=True)
        json.dump(res,open(f"res_{tag}.json","w"))
F4=['xvel','yvel','zvel','vmag']; F5=F4+['height']
job=sys.argv[1]; names=sys.argv[2].split(",")
if job=="shuf":
    lab=[(t['N'],t['D']) for t in T]; perm=rng.permutation(len(lab)); L={t['id']:lab[perm[i]] for i,t in enumerate(T)}
    run("shuf_"+sys.argv[3],F5,sys.argv[3],names,L)
else:
    run(job+"_"+sys.argv[3],F5 if sys.argv[3]=="h" else F4,job,names)
