#!/usr/bin/env python3
"""Independent numerical checks for illustrative diagrams (not performance trials)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, logit, ndtr

ROOT=Path(__file__).resolve().parents[1]

def partitions(items):
    """Enumerate set partitions for the four-node example only."""
    if not items:
        yield []
        return
    first, *rest=items
    for blocks in partitions(rest):
        yield [{first}]+[set(b) for b in blocks]
        for k in range(len(blocks)):
            out=[set(b) for b in blocks];out[k].add(first)
            yield out

def main():
    canonical=np.array([.75,.10,.10,.05])
    illustrated_native=[np.array([.75,.10,.10,.05]),
                        np.array([.10,.10,.05,.75]),
                        np.array([.10,.05,.75,.10])]
    for s,native in enumerate(illustrated_native):
        restored=np.empty(4);restored[(np.arange(4)+s)%4]=native
        assert np.allclose(restored,canonical)
    assert np.array_equal(np.array([100,100000,1e8])*np.array([1e6,1e3,1]),np.full(3,1e8))
    pi=np.array([.7,.1,.1,.1]);pj=np.array([.6,.2,.1,.1])
    assert np.isclose(pi.sum(),1) and np.isclose(pj.sum(),1)
    concentration=lambda p:1+np.sum(p*np.log(p))/np.log(len(p))
    agreement=float(np.log(4*pi@pj));ri=float(concentration(pi));rj=float(concentration(pj))
    assert abs(concentration(np.full(4,.25)))<1e-14
    weights=np.array([[0,5,4,-10],[5,0,-10,4],[4,-10,0,-10],[-10,4,-10,0.]])
    def objective(blocks):
        return sum(weights[i,j] for b in blocks for i in b for j in b if i<j)
    all_partitions=list(partitions(list(range(4))))
    assert len(all_partitions)==15
    assert objective([{0,1},{2},{3}])==5
    assert max(map(objective,all_partitions))==8
    assert objective([{0,2},{1,3}])==8
    assert weights[0,2]+weights[1,2]==-6
    assert weights[0,3]+weights[1,3]==-6
    # The motivating graph has two reports from R0 in one connected component.
    receivers={'a':0,'b':1,'c':2,'d':0}
    cluster={'a','b','c','d'}
    assert len({receivers[v] for v in cluster}) < len(cluster)
    for b in [{'a','b','c'},{'d'}]:
        assert len({receivers[v] for v in b})==len(b)
    logits=np.array([-2.,1.,3.]);scaled=logit(expit(logits/2))
    assert np.allclose(scaled,logits/2)
    report={'status':'PASS',
        'frequency_and_native_probability_examples':True,
        'illustrative_semantic_dot_product':float(pi@pj),
        'log_compatibility':agreement,'concentrations':[ri,rj],
        'weighted_semantic_feature':ri*rj*agreement,
        'uniform_semantics_neutral':True,
        'true_pair_union_bound_at_k_3_5':float(4*ndtr(-3.5)),
        'motivation_graph_receiver_conflict_and_feasible_partition':True,
        'temperature_example':True,
        'independent_bell_partition_count':len(all_partitions),
        'counterexample_greedy_objective':5,'counterexample_optimum':8,
        'scope':'Checks numerical examples and graph logic; layout also needs rendered visual inspection.'}
    target=ROOT/'audit'/'diagram_semantics.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
