#!/usr/bin/env python3
"""Deterministic mathematical and implementation checks, separate from performance data."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from experiment import Config, Scene, Pairs, simulate, features, constrained_cluster, exact_partition, matrices, cluster_metrics, UNIT_SCALE

def clone(s):
    return Scene({k:v.copy() for k,v in s.obs.items()},s.truth.copy(),s.clutter.copy(),s.positions.copy(),s.seed)

def main():
    checks={}
    maxdiff=0.
    for seed in range(20):
        s=simulate(7000000+seed,Config());p=features(s)
        t=clone(s)
        f=t.obs['frequency_value']*UNIT_SCALE[t.obs['frequency_unit']]
        t.obs['frequency_unit']=(t.obs['frequency_unit']+1)%3
        t.obs['frequency_value']=f/UNIT_SCALE[t.obs['frequency_unit']]
        pp=features(t);diff=float(np.max(np.abs(p.X-pp.X)));maxdiff=max(diff,maxdiff)
        assert np.allclose(p.X,pp.X,atol=1e-9)
        assert np.array_equal(p.gate(),pp.gate())
        # A uniform global canonical class renaming must leave dot products unchanged.
        t=clone(s); t.obs['native_semantic']=np.roll(t.obs['native_semantic'],1,axis=1)
        assert np.allclose(p.X,features(t).X,atol=1e-10)
        t=clone(s);t.obs['native_semantic'][:]=.25
        assert np.allclose(features(t).X[:,-1],0.)
        t=clone(s);t.truth=np.arange(len(t.truth))
        assert np.array_equal(p.X,features(t).X)
        # Rigid translations and rotations preserve receiver separation.
        t=clone(s);a=.7;Q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        t.positions=t.positions@Q+np.array([1000.,-400.])
        assert np.allclose(p.X,features(t).X,atol=1e-10)
        rng=np.random.default_rng(seed)
        logodds=rng.uniform(-3,3,len(p.y))
        z=constrained_cluster(s,p,expit(logodds))
        zz=constrained_cluster(s,p,expit(logodds/.7))
        assert np.array_equal(z,zz)
        m=cluster_metrics(s,p,z)
        assert m['receiver_violation']==0 and m['gate_violation']==0
    checks['independent_test_scenes']=20
    checks['max_unit_feature_absolute_difference']=maxdiff
    checks['unit_gate_invariance']=True
    checks['vocabulary_renaming_invariance']=True
    checks['missing_semantics_neutrality']=True
    checks['truth_not_used_in_features']=True
    checks['rigid_geometry_invariance']=True
    checks['temperature_partition_invariance']=True
    checks['partition_feasibility']=True
    # Abstract four-node graph: not an RF measurement or extra performance trial.
    ii,jj=np.triu_indices(4,1);w=np.array([5.,4.,-10.,-10.,4.,-10.])
    s=Scene({'receiver':np.arange(4)},np.arange(4),np.zeros(4,bool),np.zeros((4,2)),0)
    p=Pairs(np.zeros((6,6)),np.zeros((6,6)),np.zeros((6,6)),np.zeros(6,int),ii,jj,np.zeros(6),np.zeros(6),np.zeros(6,bool))
    z=constrained_cluster(s,p,expit(w));ze,opt=exact_partition(s,p,expit(w))
    obj=float(np.sum(w*(z[ii]==z[jj])))
    assert abs(obj-5.)<1e-8 and abs(opt-8.)<1e-8
    checks['counterexample']={'greedy_objective':obj,'exact_objective':float(opt),
                            'greedy_partition':z.tolist(),'exact_partition':ze.tolist()}
    out=Path(__file__).resolve().parents[1]/'audit'/'invariant_checks.json'
    out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
