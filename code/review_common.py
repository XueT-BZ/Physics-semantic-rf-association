"""Utilities for the review-driven audits. Original experiment.py is unchanged."""
from pathlib import Path
from dataclasses import replace
from itertools import combinations
import json,platform,sys,time,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import t as student_t
from experiment import Config,Scene,Pairs,simulate,features,matrices,constrained_cluster

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/review_revision'
EPS=1e-12
METRICS=['f1','precision','recall','ari','b3f1','exact','receiver_violation','gate_violation','clutter_attachment']

def parameters(rep):
    return json.loads((ROOT/f'results/linear_models_r{rep:02d}.json').read_text())

def test_scenes(rep,count=80):
    base=1_000_000+100_000*rep+20_000
    return [simulate(base+i,Config()) for i in range(count)]

def score(p,params,kind='joint'):
    q=params[kind]
    design=p.Xunweighted if kind=='unweighted' else p.X
    x=design[:,q['columns']]
    z=((x-np.asarray(q['mean']))/np.asarray(q['scale']))@np.asarray(q['coefficient'])+q['intercept']
    return expit(z/q['temperature'])

def metric(scene,p,z,k=3.5):
    """Contingency-table calculation, checked against the released metric code."""
    _,t=np.unique(scene.truth,return_inverse=True)
    _,z=np.unique(z,return_inverse=True)
    n=len(z);nt=int(t.max())+1;nc=int(z.max())+1
    cells=np.bincount(t*nc+z,minlength=nt*nc).reshape(nt,nc)
    a=cells.sum(axis=1);b=cells.sum(axis=0)
    choose=lambda x:float(np.sum(np.asarray(x,dtype=float)*(np.asarray(x,dtype=float)-1)/2))
    tp=choose(cells);actual=choose(a);predicted=choose(b)
    fp=predicted-tp;fn=actual-tp
    precision=tp/predicted if predicted else 0.
    recall=tp/actual if actual else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    total=n*(n-1)/2
    expected=actual*predicted/total if total else 0.
    denominator=.5*(actual+predicted)-expected
    ari=(tp-expected)/denominator if denominator else 1.
    bp=float(np.sum(cells*cells/b[None,:])/n)
    br=float(np.sum(cells*cells/a[:,None])/n)
    exact=[]
    for row,sz in zip(cells,a):
        if sz>=2:
            col=int(row.argmax());exact.append(row[col]==sz and b[col]==sz)
    visible=b>1
    receiver=scene.obs['receiver']
    pairs=np.unique(np.column_stack([z,receiver]),axis=0)
    distinct=np.bincount(pairs[:,0],minlength=nc)
    conflicts=visible&(distinct<b)
    same=z[p.i]==z[p.j]
    gate=p.gate(k)
    bad=int(np.sum(same&~gate));edges=int(np.sum(same))
    attached=b[z[scene.clutter]]>1
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),true_pairs=int(actual),predicted_pairs=int(predicted),
                f1=float(f1),precision=float(precision),recall=float(recall),ari=float(ari),
                b3f1=float(2*bp*br/(bp+br)),exact=float(np.mean(exact)) if exact else 1.,
                receiver_violation=float(conflicts.sum()/max(1,visible.sum())),
                gate_violation=bad/max(1,edges),receiver_conflicts=int(conflicts.sum()),gate_violations=bad,
                clutter_attachment=float(np.mean(attached)) if len(attached) else 0.,
                true_pair_gate_recall=float(np.sum(gate&p.y.astype(bool))/max(1,np.sum(p.y))))

def summary(frame,groups,columns):
    rows=[]
    for key,part in frame.groupby(groups,sort=True):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(groups,key));row['replications']=len(part)
        for name in columns:
            vals=part[name].to_numpy(dtype=float);vals=vals[np.isfinite(vals)]
            row[name]=float(np.mean(vals)) if len(vals) else None
            sd=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
            half=float(student_t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))) if len(vals)>1 else 0.
            row[name+'_lo']=row[name]-half if len(vals) else None
            row[name+'_hi']=row[name]+half if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)

def environment(task):
    import scipy,sklearn,matplotlib
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(task=task,python=sys.version,platform=platform.platform(),processor=platform.processor(),
                numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                scikit_learn=sklearn.__version__,matplotlib=matplotlib.__version__,numerical_threads=1,
                original_generator_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest())
    (OUT/f'environment_{task}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

def objective(scene,p,prob,z):
    weights=logit(np.clip(prob,EPS,1-EPS))
    return float(weights[z[p.i]==z[p.j]].sum())

def restart_greedy(scene,p,prob,starts=8):
    """Existing greedy rule with up to eight forced first merges; best objective."""
    base=constrained_cluster(scene,p,prob)
    best=base;best_value=objective(scene,p,prob,base)
    W,G=matrices(scene,p,prob);recv=scene.obs['receiver'];n=len(recv)
    allowed=G&(recv[:,None]!=recv[None,:])
    aa,bb=np.where(np.triu(allowed&(W>1e-10),1))
    order=np.lexsort((bb,aa,-W[aa,bb]))[:starts]
    for idx in order:
        a,b=int(aa[idx]),int(bb[idx])
        gains=W.copy();F=allowed.copy();np.fill_diagonal(F,False)
        active=np.ones(n,bool);clusters=[[i] for i in range(n)]
        def merge(a,b):
            clusters[a].extend(clusters[b]);clusters[b]=[]
            gains[a,:]+=gains[b,:];gains[:,a]=gains[a,:]
            F[a,:]&=F[b,:];F[:,a]=F[a,:]
            F[a,a]=False;active[b]=False
        merge(a,b)
        for _ in range(n-2):
            candidates=np.where(F&active[:,None]&active[None,:],gains,-np.inf)
            a,b=np.unravel_index(np.argmax(candidates),candidates.shape)
            if candidates[a,b]<=1e-10:break
            if a>b:a,b=b,a
            merge(a,b)
        z=np.empty(n,int)
        for q,cl in enumerate(c for c in clusters if c):z[cl]=q
        value=objective(scene,p,prob,z)
        if value>best_value+1e-10:best=z;best_value=value
    return best,best_value

def slice_scene(scene,indices):
    return Scene({k:v[indices].copy() for k,v in scene.obs.items()},scene.truth[indices].copy(),
                 scene.clutter[indices].copy(),scene.positions.copy(),scene.seed)

def save_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)
