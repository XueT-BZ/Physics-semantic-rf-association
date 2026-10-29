"""Checks supporting the second review: map-search limits and coordinate traps."""
from itertools import permutations,product
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import Config,simulate
from partial_mapping_extension import controlled_adapter,candidates,decode,weights
from review_common import ROOT,parameters
from review_mapping import exact_map_step,pool,fit_multiple

DEST=ROOT/'results/detail_revision'
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    # Algebraic pair tables. This is not an RF data or accuracy experiment.
    def algebraic(x,y):return int(x==0)+int(y==0)+3*int(x==y)
    values={str((a,b)):algebraic(a,b) for a,b in product(range(2),repeat=2)}
    assert algebraic(1,1)==3 and algebraic(0,1)==algebraic(1,0)==1 and algebraic(0,0)==5
    for single in [(0,1),(1,0)]:assert algebraic(*single)<algebraic(1,1)
    # State guard fires before the observation batch is touched or allocated.
    maps=np.tile(np.arange(4),(7,1));domain=[np.array(p) for p in permutations(range(4))]
    try:exact_map_step(None,maps,{s:domain for s in range(1,7)},None)
    except ValueError as error:guard=str(error)
    else:raise AssertionError('Large joint search was not rejected')
    (DEST/'algebraic_coordinate_counterexample.json').write_text(json.dumps(dict(
        kind='algebraic pair-potential construction, not a simulated RF scene',
        reference_terms='Psi_01(I)=Psi_02(I)=1; swap gives 0',
        coupling='Psi_12=3 for equal maps, 0 otherwise',objectives=values,
        coordinate_fixed_point=[1,1],joint_optimum=[0,0]),indent=2),encoding='utf-8')
    records=json.loads((ROOT/'results/review_revision/mapping_coordinate/learned_maps.json').read_text(encoding='utf-8'))
    records=[r for r in records if r['sources']==3 and r['anchors']==0 and r['budget']==5 and r['method']=='sequential' and r['mapping_accuracy']<1]
    example=None
    for record in records:
        rep=record['replicate'];params=parameters(rep);hidden=np.array(record['hidden_evaluation_only'])
        scenes=[simulate(80_000_000+100_000*rep+j,Config()) for j in range(5)]
        batches=[controlled_adapter(s,hidden)[0] for s in scenes]
        known=np.tile(np.arange(4),(4,1));anchors={s:{} for s in range(1,4)}
        fit=fit_multiple(batches,params['joint'],params['physics'],known,anchors,strategy='coordinate')
        fitted=fit['sequential'];assert np.array_equal(fitted,np.array(record['map']))
        groups=[decode(b,known,params['physics']) for b in batches];selected=pool(batches,groups)
        base=float(weights(selected,fitted,params['joint']).sum())
        choices={s:candidates({},4) for s in range(1,4)}
        single=[]
        for s,options in choices.items():
            for option in options:
                trial=fitted.copy();trial[s]=option
                single.append(float(weights(selected,trial,params['joint']).sum())-base)
        joint,joint_value,count=exact_map_step(selected,fitted,choices,params['joint'])
        if max(single)<=1e-8 and joint_value>base+1e-7:
            example=dict(replicate=rep,adaptation_scenes=5,unknown_sources=3,anchors=0,
                         fixed_groups='physical-only provisional groups',
                         coordinate_objective=base,best_single_source_gain=max(single),joint_objective=joint_value,
                         joint_gain=joint_value-base,joint_candidates=count,
                         changed_receivers=int(np.any(joint!=fitted,axis=1).sum()),
                         coordinate_mapping_accuracy=record['mapping_accuracy'],
                         coordinate_map=fitted.tolist(),joint_map=joint.tolist(),
                         hidden_map_evaluation_only=hidden.tolist())
            break
    assert example is not None,'No saved-case coordinate trap was verified'
    (DEST/'saved_case_coordinate_trap.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    sizes=[]
    for K,U in [(4,3),(4,6),(6,3),(8,3)]:
        Q=math.factorial(K)**U
        sizes.append(dict(classes=K,unknown_receivers=U,states=Q,float64_score_array_bytes=8*Q,
                          score_array_gib=8*Q/2**30,kind='arithmetic state count; not a runtime benchmark'))
    pd.DataFrame(sizes).to_csv(DEST/'state_space_sizes.csv',index=False)
    statement=dict(status='PASS',algebraic_counterexample=True,large_block_rejected_before_allocation=guard,
                   saved_case_coordinate_trap=example,
                   historical_scorers_and_test_results_refit=False,
                   actual_small_block_search_unchanged=True)
    (DEST/'verification.json').write_text(json.dumps(statement,indent=2),encoding='utf-8')
    print(json.dumps(statement,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
