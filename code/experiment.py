#!/usr/bin/env python3
"""Reproducible observation-to-event association. All RF data are synthetic.

Run from any directory: python code/experiment.py --out results --seeds 10
No I/Q recordings, public RF datasets, or pre-trained models are used.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar, milp, Bounds, LinearConstraint, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.stats import t as student_t
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, adjusted_rand_score)
from threadpoolctl import threadpool_limits

C = 299792458.0
EPS = 1e-12
K = 4
UNIT_SCALE = np.array([1.0, 1e3, 1e6])
FEATURE_NAMES = ['log1p_abs_time_z', 'log1p_excess_time_z', 'log1p_freq_z',
                 'log1p_bandwidth_z', 'log1p_duration_z', 'soft_semantic_evidence']

@dataclass(frozen=True)
class Config:
    receivers: int = 4
    events: int = 14
    window_us: float = 120.0
    detection: float = .8
    clutter: float = .8  # Poisson mean per receiver and window
    collisions: float = .25
    semantic_error: float = .12
    semantic_missing: float = .1
    clock_us: float = .8  # actual residual, common to one receiver within a scene
    declared_clock_us: float = .8
    freq_bias_khz: float = .25
    declared_freq_khz: float = .25
    multipath_probability: float = 0.0
    multipath_mean_us: float = 8.0
    heavy_tail: bool = False
    corrupted_receiver: bool = False

@dataclass
class Scene:
    obs: dict[str, np.ndarray]
    truth: np.ndarray
    clutter: np.ndarray
    positions: np.ndarray
    seed: int

@dataclass
class Pairs:
    X: np.ndarray
    Xraw: np.ndarray
    Xunweighted: np.ndarray
    y: np.ndarray
    i: np.ndarray
    j: np.ndarray
    time_z: np.ndarray
    freq_z: np.ndarray
    unseen: np.ndarray

    def gate(self, k: float = 3.5) -> np.ndarray:
        return (self.time_z <= k) & (self.freq_z <= k)


def simulate(seed: int, cfg: Config) -> Scene:
    """Independent complete scene; never sample a pair label and then its features."""
    rng = np.random.default_rng(seed)
    S, E = cfg.receivers, cfg.events
    # Receivers lie on a 2-km radius ring, with scene-specific azimuth rotation.
    angles = np.arange(S)*2*np.pi/S + rng.uniform(0, 2*np.pi)
    pos = 2000*np.column_stack([np.cos(angles), np.sin(angles)])
    b = rng.normal(0, cfg.clock_us*1e-6, S)
    d = rng.normal(0, cfg.freq_bias_khz*1e3, S)
    tau = rng.uniform(0, cfg.window_us*1e-6, E)
    carrier = 100e6 + rng.integers(0, 6, E)*1800 + rng.normal(0, 180, E)
    bw = rng.uniform(20e3, 60e3, E)
    dur = rng.uniform(20e-6, 80e-6, E)
    loc = rng.uniform(-1500, 1500, (E,2))
    protocol = rng.integers(0, K, E)
    # Different events can be nearly coincident in all four physical attributes.
    for e in range(1, E):
        if rng.random() < cfg.collisions:
            a = rng.integers(0,e)
            tau[e] = tau[a] + rng.normal(0,1.5e-6)
            carrier[e] = carrier[a] + rng.normal(0,150)
            bw[e] = max(1e3,bw[a]+rng.normal(0,2e3))
            dur[e] = max(1e-6,dur[a]+rng.normal(0,2e-6))
    rows, truths, noise_labels = [], [], []
    for s in range(S):
        st = (.6+.4*s)*1e-6
        sf = (.12+.08*s)*1e3
        sb = (1.5+.3*s)*1e3
        sd = (1.0+.3*s)*1e-6
        ids = list(np.flatnonzero(rng.random(E) < cfg.detection))
        false_count = rng.poisson(cfg.clutter)
        for local, e in enumerate(ids + [-1]*false_count):
            false = e < 0
            if false:
                tx, fx, wx, lx = (rng.uniform(0,cfg.window_us*1e-6),
                                  100e6+rng.integers(0,6)*1800+rng.normal(0,180),
                                  rng.uniform(20e3,60e3),rng.uniform(20e-6,80e-6))
                q = int(rng.integers(K)); delay=0.
            else:
                tx,fx,wx,lx,q=tau[e],carrier[e],bw[e],dur[e],int(protocol[e])
                delay=np.linalg.norm(loc[e]-pos[s])/C
            jitter=(rng.standard_t(3)/np.sqrt(3) if cfg.heavy_tail else rng.normal())*st
            excess=(rng.exponential(cfg.multipath_mean_us*1e-6)
                    if rng.random()<cfg.multipath_probability else 0.)
            t=tx+delay+b[s]+jitter+excess
            f=fx+d[s]+rng.normal(0,sf)
            w=max(1.,wx+rng.normal(0,sb)); length=max(1e-9,lx+rng.normal(0,sd))
            missing=rng.random()<cfg.semantic_missing
            wrong=rng.random()<cfg.semantic_error
            qhat=int((q+rng.integers(1,K))%K) if wrong else q
            conf=rng.uniform(.45,.9) if wrong else rng.uniform(.6,.95)
            if cfg.corrupted_receiver and s==S-1:
                qhat=(q+1)%K; conf=.99; missing=False
            probs=np.full(K,(1-conf)/(K-1));probs[qhat]=conf
            if missing: probs[:]=1/K
            # Native vocabulary column a denotes canonical family (a+s)%K.
            native=probs[(np.arange(K)+s)%K]
            u=int(rng.integers(3))
            rows.append([s,t,f/UNIT_SCALE[u],u,w,length,st,sf,sb,sd,
                         cfg.declared_clock_us*1e-6,cfg.declared_freq_khz*1e3,*native])
            truths.append(int(e) if not false else E+len(truths))
            noise_labels.append(false)
    # Shuffling prevents exploiting ordering by receiver/event.
    order=rng.permutation(len(rows))
    a=np.array(rows,dtype=float)[order]
    names=['receiver','time_s','frequency_value','frequency_unit','bandwidth_hz',
           'duration_s','sigma_time_s','sigma_frequency_hz','sigma_bandwidth_hz',
           'sigma_duration_s','declared_clock_s','declared_freq_bias_hz']
    obs={name:a[:,k] for k,name in enumerate(names)}
    obs['native_semantic']=a[:,12:16]
    obs['receiver']=obs['receiver'].astype(int)
    obs['frequency_unit']=obs['frequency_unit'].astype(int)
    return Scene(obs,np.array(truths)[order],np.array(noise_labels)[order],pos,seed)


def features(scene: Scene) -> Pairs:
    o=scene.obs;s=o['receiver'];N=len(s)
    i,j=np.triu_indices(N,1);keep=s[i]!=s[j];i,j=i[keep],j[keep]
    # Truth is only attached AFTER the observable feature computations.
    f=o['frequency_value']*UNIT_SCALE[o['frequency_unit']]
    dt=np.abs(o['time_s'][i]-o['time_s'][j])
    geo=np.linalg.norm(scene.positions[s[i]]-scene.positions[s[j]],axis=1)/C
    ut=np.sqrt(o['sigma_time_s'][i]**2+o['sigma_time_s'][j]**2+
               o['declared_clock_s'][i]**2+o['declared_clock_s'][j]**2)
    uf=np.sqrt(o['sigma_frequency_hz'][i]**2+o['sigma_frequency_hz'][j]**2+
               o['declared_freq_bias_hz'][i]**2+o['declared_freq_bias_hz'][j]**2)
    ub=np.hypot(o['sigma_bandwidth_hz'][i],o['sigma_bandwidth_hz'][j])
    ud=np.hypot(o['sigma_duration_s'][i],o['sigma_duration_s'][j])
    tz=np.maximum(0,dt-geo)/ut;fz=np.abs(f[i]-f[j])/uf
    v=np.column_stack([dt/ut,tz,fz,
            np.abs(o['bandwidth_hz'][i]-o['bandwidth_hz'][j])/ub,
            np.abs(o['duration_s'][i]-o['duration_s'][j])/ud])
    phys=np.log1p(v)
    native=o['native_semantic'];pi=np.empty_like(native)
    for n in range(N): pi[n,(np.arange(K)+s[n])%K]=native[n]
    reliability=1+np.sum(pi*np.log(np.maximum(pi,EPS)),axis=1)/np.log(K)
    evidence=np.log(np.maximum(K*np.sum(pi[i]*pi[j],axis=1),EPS))
    sem=reliability[i]*reliability[j]*evidence
    X=np.column_stack([phys,sem])
    raw=X.copy();raw[:,2]=np.log1p(np.abs(o['frequency_value'][i]-o['frequency_value'][j])/uf)
    unweighted=np.column_stack([phys,evidence])
    return Pairs(X,raw,unweighted,(scene.truth[i]==scene.truth[j]).astype(int),
                 i,j,tz,fz,(s[i]>=4)|(s[j]>=4))


class Scorer:
    def __init__(self, kind: str):
        self.kind=kind; self.temperature=1.
        self.cols={'time':[0,1],'frequency':[2],'semantic':[5],
                   'physics':[0,1,2,3,4]}.get(kind,list(range(6)))
        self.model=(HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,
                    learning_rate=.08,l2_regularization=1.,random_state=123,
                    early_stopping=False) if kind=='boosted' else
                    make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500)))
    def design(self,p:Pairs)->np.ndarray:
        X=p.Xraw if self.kind=='raw_units' else (p.Xunweighted if self.kind=='unweighted' else p.X)
        return X[:,self.cols]
    def fit(self,pairs:list[Pairs])->None:
        X=np.concatenate([self.design(p) for p in pairs]);y=np.concatenate([p.y for p in pairs])
        self.model.fit(X,y)
    def logits(self,p:Pairs)->np.ndarray:
        X=self.design(p)
        if self.kind=='boosted':
            return logit(np.clip(self.model.predict_proba(X)[:,1],EPS,1-EPS))
        return self.model.decision_function(X)
    def calibrate(self,pairs:list[Pairs])->None:
        zz=np.concatenate([self.logits(p)[p.gate()] for p in pairs])
        yy=np.concatenate([p.y[p.gate()] for p in pairs])
        def objective(logT:float)->float:
            z=zz/np.exp(logT)
            return float(np.mean(np.logaddexp(0,z)-yy*z))
        res=minimize_scalar(objective,bounds=(np.log(.1),np.log(10.)),method='bounded',
                            options={'xatol':1e-7})
        if not res.success:raise RuntimeError('Temperature optimization failed')
        self.temperature=float(np.exp(res.x))
    def score(self,p:Pairs,calibrated=False,gated=False,k=3.5)->np.ndarray:
        prob=expit(self.logits(p)/(self.temperature if calibrated else 1.))
        if gated:prob=np.where(p.gate(k),prob,0.)
        return prob


def expected_calibration_error(y:np.ndarray,p:np.ndarray,bins:int=10)->float:
    idx=np.minimum((p*bins).astype(int),bins-1)
    return float(sum(np.mean(idx==b)*abs(np.mean(y[idx==b])-np.mean(p[idx==b]))
                     for b in range(bins) if np.any(idx==b)))


def pair_metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    pp=np.clip(p,EPS,1-EPS)
    return dict(auroc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)),nll=float(log_loss(y,pp,labels=[0,1])),
                ece=expected_calibration_error(y,p),positive_rate=float(np.mean(y)))


def labels_from_clusters(clusters:list[list[int]],N:int)->np.ndarray:
    z=np.empty(N,int)
    for k,cc in enumerate(clusters):z[cc]=k
    return z


def matrices(scene:Scene,pairs:Pairs,prob:np.ndarray,k:float=3.5):
    n=len(scene.truth);W=np.zeros((n,n));G=np.zeros((n,n),bool)
    w=logit(np.clip(prob,EPS,1-EPS))
    W[pairs.i,pairs.j]=w;W[pairs.j,pairs.i]=w
    G[pairs.i,pairs.j]=pairs.gate(k);G[pairs.j,pairs.i]=pairs.gate(k)
    return W,G


def constrained_cluster(scene:Scene,pairs:Pairs,prob:np.ndarray,*,
                        exclude_receiver=True,complete_gate=True,k=3.5,
                        tau=.5)->np.ndarray:
    """Greedy positive-gain agglomeration; not a globally optimal solver.

    Dense matrix implementation: O(N^3) comparison work, O(N^2) storage.
    A within-receiver edge has weight zero if that constraint is ablated.
    """
    n=len(scene.truth);W,G=matrices(scene,pairs,prob,k)
    if tau!=.5:
        W[pairs.i,pairs.j]-=logit(tau);W[pairs.j,pairs.i]-=logit(tau)
    recv=scene.obs['receiver'];disjoint=recv[:,None]!=recv[None,:]
    F=np.ones((n,n),bool)
    if complete_gate:F &= G | ~disjoint  # receiver constraint is separate
    if exclude_receiver:F &= disjoint
    np.fill_diagonal(F,False)
    active=np.ones(n,bool);clusters=[[a] for a in range(n)]
    gain=W.copy()
    for _ in range(n-1):
        candidates=np.where(F & active[:,None] & active[None,:],gain,-np.inf)
        a,b=np.unravel_index(np.argmax(candidates),candidates.shape)
        if candidates[a,b]<=1e-10:break
        if a>b:a,b=b,a
        clusters[a].extend(clusters[b]);clusters[b]=[]
        gain[a,:]+=gain[b,:];gain[:,a]=gain[a,:]
        F[a,:]&=F[b,:];F[:,a]=F[a,:]
        F[a,a]=False;active[b]=False
    return labels_from_clusters([cl for cl in clusters if cl],n)


def connected_cluster(scene:Scene,p:Pairs,prob:np.ndarray,assignment=False)->np.ndarray:
    n=len(scene.truth);parent=list(range(n))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    if not assignment:
        for a,b in zip(p.i[(prob>.5)&p.gate()],p.j[(prob>.5)&p.gate()]):union(int(a),int(b))
    else:
        W,G=matrices(scene,p,prob);ss=scene.obs['receiver']
        for r,t in combinations(np.unique(ss),2):
            aa=np.flatnonzero(ss==r);bb=np.flatnonzero(ss==t)
            # Dummy columns implement optional/unmatched assignments at score 0.
            gain=np.zeros((len(aa),len(bb)+len(aa)))
            gain[:,:len(bb)]=np.where(G[np.ix_(aa,bb)],W[np.ix_(aa,bb)],-1e6)
            rr,cc=linear_sum_assignment(gain,maximize=True)
            for a,b in zip(rr,cc):
                if b<len(bb) and gain[a,b]>0:union(int(aa[a]),int(bb[b]))
    return np.array([find(a) for a in range(n)])


def cluster_metrics(scene:Scene,p:Pairs,z:np.ndarray)->dict[str,float]:
    truth=scene.truth;n=len(z);i,j=np.triu_indices(n,1)
    yp=z[i]==z[j];yt=truth[i]==truth[j]
    tp=np.sum(yp&yt);fp=np.sum(yp&~yt);fn=np.sum(~yp&yt)
    precision=float(tp/(tp+fp)) if tp+fp else 0.
    recall=float(tp/(tp+fn)) if tp+fn else 0.
    f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.
    pred_sets=[set(np.flatnonzero(z==lab)) for lab in np.unique(z)]
    gold_sets=[set(np.flatnonzero(truth==lab)) for lab in np.unique(truth)]
    bprec=[];brec=[]
    for a in range(n):
        overlap=np.sum((z==z[a])&(truth==truth[a]))
        bprec.append(overlap/np.sum(z==z[a]));brec.append(overlap/np.sum(truth==truth[a]))
    bp,br=np.mean(bprec),np.mean(brec)
    visible=[g for g in gold_sets if len(g)>=2]
    exact=np.mean([g in pred_sets for g in visible]) if visible else 1.
    nonsingle=[cc for cc in pred_sets if len(cc)>1]
    invalid=np.mean([len(set(scene.obs['receiver'][list(cc)]))<len(cc)
                     for cc in nonsingle]) if nonsingle else 0.
    same=z[p.i]==z[p.j]
    gate_viol=float(np.sum(same&~p.gate())/max(1,np.sum(same)))
    attached=[np.sum(z==z[a])>1 for a in np.flatnonzero(scene.clutter)]
    return dict(f1=f1,precision=precision,recall=recall,ari=float(adjusted_rand_score(truth,z)),
                b3f1=float(2*bp*br/(bp+br)),exact=float(exact),receiver_violation=float(invalid),
                gate_violation=gate_viol,clutter_attachment=float(np.mean(attached)) if attached else 0.)


def exact_partition(scene:Scene,p:Pairs,prob:np.ndarray):
    """Binary transitivity formulation, solved only for small instances by HiGHS."""
    n=len(scene.truth);edges=list(combinations(range(n),2));m=len(edges)
    idx={e:k for k,e in enumerate(edges)};W,G=matrices(scene,p,prob)
    c=-np.array([W[a,b] for a,b in edges]);ub=np.array([G[a,b] for a,b in edges],float)
    rr=[];cc=[];vv=[];row=0
    for a,b,c0 in combinations(range(n),3):
        inds=[idx[(a,b)],idx[(a,c0)],idx[(b,c0)]]
        for neg in range(3):
            for d,index in enumerate(inds):rr.append(row);cc.append(index);vv.append(-1. if d==neg else 1.)
            row+=1
    A=coo_matrix((vv,(rr,cc)),shape=(row,m)).tocsc()
    result=milp(c,integrality=np.ones(m),bounds=Bounds(np.zeros(m),ub),
                constraints=LinearConstraint(A,-np.inf,np.ones(row)),
                options={'time_limit':20.,'mip_rel_gap':0.})
    if not result.success:raise RuntimeError(f'Exact verification not optimal: {result.message}')
    parent=list(range(n))
    def root(a):
        while parent[a]!=a:a=parent[a]
        return a
    for value,(a,b) in zip(result.x,edges):
        if value>.5:parent[root(b)]=root(a)
    return np.array([root(a) for a in range(n)]),float(-result.fun)


def scene_csv(scene:Scene,seed:int,split:str,index:int)->pd.DataFrame:
    df=pd.DataFrame({k:v for k,v in scene.obs.items() if v.ndim==1})
    for q in range(K):df[f'native_semantic_{q}']=scene.obs['native_semantic'][:,q]
    df['truth_event']=scene.truth;df['is_clutter']=scene.clutter
    df['replicate']=seed;df['split']=split;df['scene']=index;df['scene_seed']=scene.seed
    df['receiver_x_m']=scene.positions[scene.obs['receiver'],0]
    df['receiver_y_m']=scene.positions[scene.obs['receiver'],1]
    return df


def summarize(df:pd.DataFrame,group:list[str],metrics:list[str])->pd.DataFrame:
    rows=[]
    for key,part in df.groupby(group,sort=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(group,key));row['n_replicates']=len(part)
        for m in metrics:
            arr=part[m].to_numpy();mean=float(np.mean(arr));sd=float(np.std(arr,ddof=1)) if len(arr)>1 else 0.
            half=float(student_t.ppf(.975,len(arr)-1)*sd/np.sqrt(len(arr))) if len(arr)>1 else 0.
            row[m]=mean;row[m+'_sd']=sd;row[m+'_lo']=mean-half;row[m+'_hi']=mean+half
        rows.append(row)
    return pd.DataFrame(rows)


def run(out:Path,seeds:int=10,quick=False,first:int=0):
    out.mkdir(parents=True,exist_ok=True);started=time.time()
    ntrain,ncal,ntest=(25,10,12) if quick else (120,40,80)
    nsweep=4 if quick else 20
    kinds=['time','frequency','semantic','physics','joint','boosted','raw_units','unweighted']
    prows=[];crows=[];srows=[];grow=[];exactrows=[];manifest=[];paramrows=[]
    nominal=Config();pdf_saved=False
    for rep in range(first,first+seeds):
        root=1000000+rep*100000
        splitdata={};pairdata={}
        for split,offset,num in [('train',0,ntrain),('calibration',10000,ncal),('test',20000,ntest)]:
            scenes=[simulate(root+offset+k,nominal) for k in range(num)]
            pairs=[features(sc) for sc in scenes];splitdata[split]=scenes;pairdata[split]=pairs
            # All split observations are saved; no network downloads are needed.
            pd.concat([scene_csv(sc,rep,split,k) for k,sc in enumerate(scenes)],ignore_index=True).to_csv(
                out/f'observations_r{rep:02d}_{split}.csv.gz',index=False,compression='gzip')
            manifest.append(dict(replicate=rep,split=split,first_scene_seed=root+offset,scenes=num,
                   observations=sum(len(sc.truth) for sc in scenes),pairs=sum(len(p.y) for p in pairs),
                   positive_pairs=sum(int(p.y.sum()) for p in pairs)))
        models={kind:Scorer(kind) for kind in kinds}
        for kind,m in models.items():
            m.fit(pairdata['train']);m.calibrate(pairdata['calibration'])
            paramrows.append(dict(replicate=rep,method=kind,temperature=m.temperature))
        modelmeta={}
        for kind,m in models.items():
            if kind!='boosted':
                scaler,lr=m.model.steps[0][1],m.model.steps[1][1]
                modelmeta[kind]=dict(columns=m.cols,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                      coefficient=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),temperature=m.temperature)
        (out/f'linear_models_r{rep:02d}.json').write_text(json.dumps(modelmeta,indent=2))
        variants=[(k,k,False,False) for k in kinds]+[
                 ('joint_gated','joint',False,True),('joint_calibrated','joint',True,True),
                 ('physics_calibrated','physics',True,True),('boosted_calibrated','boosted',True,True)]
        y=np.concatenate([p.y for p in pairdata['test']]);gate=np.concatenate([p.gate() for p in pairdata['test']])
        for name,kind,cal,gating in variants:
            pred=np.concatenate([models[kind].score(p,cal,gating) for p in pairdata['test']])
            row=dict(replicate=rep,method=name,**pair_metrics(y,pred))
            row['admitted_ece']=expected_calibration_error(y[gate],pred[gate]);prows.append(row)
            if rep==0:
                np.savez_compressed(out/f'prediction_{name}_r00.npz',y=y,prob=pred,gate=gate)
        for k,(scene,p) in enumerate(zip(splitdata['test'],pairdata['test'])):
            jp=models['joint'].score(p,True,False)
            pp=models['physics'].score(p,True,False)
            hp=models['boosted'].score(p,True,False)
            variants_z={
                'physics_constrained':lambda:constrained_cluster(scene,p,pp),
                'boosted_constrained':lambda:constrained_cluster(scene,p,hp),
                'joint_components':lambda:connected_cluster(scene,p,jp),
                'pairwise_assignment':lambda:connected_cluster(scene,p,jp,True),
                'joint_constrained':lambda:constrained_cluster(scene,p,jp),
                'no_receiver_exclusion':lambda:constrained_cluster(scene,p,jp,exclude_receiver=False),
                'no_complete_gate':lambda:constrained_cluster(scene,p,jp,complete_gate=False),
                'unweighted_semantics':lambda:constrained_cluster(scene,p,models['unweighted'].score(p,True,False)),
            }
            for name,fn in variants_z.items():
                t0=time.perf_counter();z=fn();elapsed=time.perf_counter()-t0
                metric=cluster_metrics(scene,p,z)
                if name=='joint_constrained':
                    assert metric['receiver_violation']==0 and metric['gate_violation']==0
                    zr=constrained_cluster(scene,p,models['joint'].score(p,False,False))
                    assert np.array_equal(z[:,None]==z[None,:],zr[:,None]==zr[None,:]),'Temperature invariance failed'
                crows.append(dict(replicate=rep,scene=k,method=name,observations=len(scene.truth),
                                  runtime_ms=elapsed*1000,**metric))
            if rep==0 and k==0:
                np.savez_compressed(out/'example_scene.npz',truth=scene.truth,receiver=scene.obs['receiver'],
                    predicted=constrained_cluster(scene,p,jp),time_us=scene.obs['time_s']*1e6,
                    freq_khz=scene.obs['frequency_value']*UNIT_SCALE[scene.obs['frequency_unit']]/1e3,
                    positions=scene.positions)
        # Each condition is a new complete-scene simulation with the fitted models frozen.
        scenarios={
            'nominal':nominal,
            'dense':replace(nominal,events=28),
            'missed_detections':replace(nominal,detection=.55),
            'clutter':replace(nominal,clutter=2.5),
            'semantic_error':replace(nominal,semantic_error=.45),
            'semantic_missing':replace(nominal,semantic_missing=.65),
            'clock_misspecified':replace(nominal,clock_us=6.),
            'frequency_misspecified':replace(nominal,freq_bias_khz=1.5),
            'multipath':replace(nominal,multipath_probability=.2),
            'heavy_tails':replace(nominal,heavy_tail=True),
            'unseen_receiver':replace(nominal,receivers=5),
            'overconfident_wrong':replace(nominal,corrupted_receiver=True)}
        sweeps=[]
        for name,cfg in scenarios.items():sweeps.append(('scenario',name,cfg))
        for err in [0.,.1,.25,.4,.6,.75]:sweeps.append(('semantic',str(err),replace(nominal,semantic_error=err)))
        for clk in [0.,.8,2.,5.,10.,20.]:
            sweeps.append(('clock_underreported',str(clk),replace(nominal,clock_us=clk)))
            sweeps.append(('clock_matched',str(clk),replace(nominal,clock_us=clk,declared_clock_us=clk)))
        for E in [8,16,32,64]:sweeps.append(('scaling',str(E),replace(nominal,events=E)))
        for group,value,cfg in sweeps:
            # Same root per condition provides common random numbers where applicable.
            acc={'physics':[],'joint':[]};py=[];preds={'physics':[],'joint':[]};unseen_y=[];unseen_pred=[]
            for kk in range(nsweep):
                sc=simulate(root+30000+kk,cfg);p=features(sc);py.append(p.y)
                for kind in acc:
                    prob=models[kind].score(p,True,False)
                    t0=time.perf_counter();z=constrained_cluster(sc,p,prob);rt=time.perf_counter()-t0
                    acc[kind].append(dict(**cluster_metrics(sc,p,z),runtime_ms=rt*1000,
                        observations=len(sc.truth),gate_recall=float(p.gate()[p.y==1].mean()),
                        retained_fraction=float(p.gate().mean())))
                    preds[kind].append(np.where(p.gate(),prob,0.))
                    if kind=='joint' and p.unseen.any():
                        unseen_y.append(p.y[p.unseen]);unseen_pred.append(np.where(p.gate(),prob,0.)[p.unseen])
            for kind,rows in acc.items():
                row=dict(replicate=rep,group=group,value=value,method=kind,**pd.DataFrame(rows).mean().to_dict(),
                         **pair_metrics(np.concatenate(py),np.concatenate(preds[kind])))
                if kind=='joint' and unseen_y:row['novel_receiver_ap']=average_precision_score(np.concatenate(unseen_y),np.concatenate(unseen_pred))
                srows.append(row)
        for width in [1.,2.,3.,3.5,5.]:
            values=[]
            for sc,p in zip(splitdata['test'][:nsweep],pairdata['test'][:nsweep]):
                prob=models['joint'].score(p,True,False)
                z=constrained_cluster(sc,p,prob,k=width)
                values.append(dict(f1=cluster_metrics(sc,p,z)['f1'],
                    gate_recall=float(p.gate(width)[p.y==1].mean()),retained_fraction=float(p.gate(width).mean())))
            grow.append(dict(replicate=rep,k=width,**pd.DataFrame(values).mean().to_dict()))
        # Small-scene exact optimization is a verification test, not the large-scale method.
        for kk in range(2 if quick else 6):
            cfg=replace(nominal,events=4,clutter=.3,collisions=.5,window_us=35.)
            sc=simulate(root+50000+kk,cfg);p=features(sc);prob=models['joint'].score(p,True,False)
            zg=constrained_cluster(sc,p,prob);t0=time.perf_counter();ze,opt=exact_partition(sc,p,prob);rt=time.perf_counter()-t0
            W,_=matrices(sc,p,prob);ii,jj=np.triu_indices(len(zg),1)
            obj=float(np.sum(W[ii,jj]*(zg[ii]==zg[jj])));assert opt>=obj-1e-6
            gap=(opt-obj)/max(abs(opt),1e-9)
            exactrows.append(dict(replicate=rep,case=kk,n=len(zg),greedy_objective=obj,optimal_objective=opt,
                   relative_gap=gap,greedy_f1=cluster_metrics(sc,p,zg)['f1'],exact_f1=cluster_metrics(sc,p,ze)['f1'],exact_ms=rt*1000))
        print(f'Replicate {rep+1}/{seeds} complete; elapsed {time.time()-started:.1f}s',flush=True)
        # Checkpoint only computed results; final summaries use all completed replicates.
        pd.DataFrame(prows).to_csv(out/'pair_per_replicate.csv',index=False)
        pd.DataFrame(crows).to_csv(out/'cluster_per_scene.csv',index=False)
        pd.DataFrame(srows).to_csv(out/'stress_per_replicate.csv',index=False)
        pd.DataFrame(grow).to_csv(out/'gate_per_replicate.csv',index=False)
        pd.DataFrame(exactrows).to_csv(out/'exact_cases.csv',index=False)
    pm=['auroc','ap','brier','nll','ece','admitted_ece','positive_rate']
    summarize(pd.DataFrame(prows),['method'],pm).to_csv(out/'pair_summary.csv',index=False)
    cm=['f1','precision','recall','ari','b3f1','exact','receiver_violation','gate_violation','clutter_attachment','runtime_ms']
    cd=pd.DataFrame(crows).groupby(['replicate','method'],as_index=False)[cm].mean()
    cd.to_csv(out/'cluster_per_replicate.csv',index=False)
    summarize(cd,['method'],cm).to_csv(out/'cluster_summary.csv',index=False)
    sm=['f1','precision','recall','ari','exact','runtime_ms','observations','gate_recall','retained_fraction','auroc','ap','brier','nll']
    summarize(pd.DataFrame(srows),['group','value','method'],sm).to_csv(out/'stress_summary.csv',index=False)
    summarize(pd.DataFrame(grow),['k'],['f1','gate_recall','retained_fraction']).to_csv(out/'gate_summary.csv',index=False)
    pd.DataFrame(paramrows).to_csv(out/'temperatures.csv',index=False)
    pd.DataFrame(manifest).to_csv(out/'split_manifest.csv',index=False)
    # Paired intervals across independent full training/calibration/test replications.
    delta=[]
    for group,col,base,target,metric in [('pair','method','physics_calibrated','joint_calibrated','ap'),
                ('pair','method','joint_gated','joint_calibrated','brier'),
                ('cluster','method','physics_constrained','joint_constrained','f1'),
                ('cluster','method','joint_components','joint_constrained','f1'),
                ('cluster','method','boosted_constrained','joint_constrained','f1')]:
        dd=pd.DataFrame(prows) if group=='pair' else cd
        wide=dd.pivot(index='replicate',columns=col,values=metric)
        v=wide[target]-wide[base];se=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.
        h=float(student_t.ppf(.975,len(v)-1)*se) if len(v)>1 else 0.
        delta.append(dict(group=group,metric=metric,baseline=base,target=target,delta=float(v.mean()),
                          lo=float(v.mean()-h),hi=float(v.mean()+h)))
    pd.DataFrame(delta).to_csv(out/'paired_differences.csv',index=False)
    env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
         'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'pandas':pd.__version__,
         'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'thread_limit':1,
         'elapsed_seconds':time.time()-started,'replications':seeds,'configuration':asdict(nominal),
         'train_scenes_per_rep':ntrain,'calibration_scenes_per_rep':ncal,'test_scenes_per_rep':ntest,
         'stress_scenes_per_condition_per_rep':nsweep,'generator':'synthetic descriptors, not IQ waveform simulation',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if Path('/proc/cpuinfo').exists():
        env['cpu']=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),env['cpu'])
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print('FINISHED',json.dumps(env,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--first',type=int,default=0)
    args=parser.parse_args()
    if args.seeds<1:parser.error('--seeds must be positive')
    with threadpool_limits(limits=1):run(args.out,args.seeds,args.quick,args.first)
