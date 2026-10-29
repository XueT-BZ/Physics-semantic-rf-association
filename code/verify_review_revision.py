"""Independent checks for conditional optimization and saved scientific claims."""
from dataclasses import replace
from itertools import product,combinations
import json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from experiment import Config,Scene,Pairs,simulate,features,constrained_cluster
from partial_mapping_extension import controlled_adapter,candidates,decode,weights
from review_common import ROOT,OUT,parameters,metric,objective,restart_greedy
from review_mapping import exact_map_step,pool,validate
from review_risks import false_alarms,fragments,recall_ceiling

def main():
    checks={}
    params=parameters(0);scene=simulate(80_000_000,Config())
    hidden=np.array([[0,1,2,3],[2,0,3,1],[3,2,1,0],[0,1,2,3]])
    batch=controlled_adapter(scene,hidden)[0]
    initial=np.tile(np.arange(4),(4,1));groups=decode(batch,initial,params['physics'])
    selected=pool([batch],[groups]);choices={1:candidates({},4),2:candidates({},4)}
    fitted,value,count=exact_map_step(selected,initial,choices,params['joint'])
    brute=-np.inf
    for a,b in product(choices[1],choices[2]):
        trial=initial.copy();trial[1]=a;trial[2]=b
        brute=max(brute,float(weights(selected,trial,params['joint']).sum()))
    assert abs(brute-value)<1e-7 and count==576
    checks['conditional_pair_tables_match_direct_576_state_enumeration']=True
    # Event references are evaluation-only. Altering them cannot change the adapter.
    changed=replace(scene,truth=np.arange(len(scene.truth))+100000,clutter=~scene.clutter)
    other=controlled_adapter(changed,hidden)[0]
    for name in batch.__dataclass_fields__:
        assert np.array_equal(getattr(batch,name),getattr(other,name))
    checks['observable_adapter_is_independent_of_reference_labels']=True
    # Algebraic example from the main figure.
    ii,jj=np.asarray(list(combinations(range(4),2))).T
    w=np.array([5.,4.,-10.,-10.,4.,-10.]);n=len(ii)
    p=Pairs(np.zeros((n,6)),np.zeros((n,6)),np.zeros((n,6)),np.zeros(n),ii,jj,np.zeros(n),np.zeros(n),np.zeros(n))
    s=Scene({'receiver':np.arange(4)},np.arange(4),np.zeros(4,bool),np.zeros((4,2)),0)
    prob=expit(w);z=constrained_cluster(s,p,prob);zr,jr=restart_greedy(s,p,prob)
    assert abs(objective(s,p,prob,z)-5)<1e-7 and abs(jr-8)<1e-7
    checks['restart_recovers_four_node_score_8_instead_of_5']=True
    for shared in [False,True]:
        a=false_alarms(scene,86_000_123,shared=shared)
        assert len(a.truth)==len(scene.truth)+12 and a.clutter.sum()==scene.clutter.sum()+12
        assert len(np.unique(a.truth[-12:]))==12
    frag=fragments(scene,86_000_124);p=features(frag)
    from review_common import score
    z=constrained_cluster(frag,p,score(p,params))
    assert metric(frag,p,z)['recall']<=recall_ceiling(frag)+1e-12
    checks['matched_alarm_counts_and_fragment_recall_bound']=True
    nominal={1_000_000+100_000*r+off+j for r in range(10) for off,m in [(0,120),(10000,40),(20000,80)] for j in range(m)}
    adaptation={80_000_000+100_000*r+j for r in range(10) for j in range(40)}
    assert len(adaptation)==400 and not adaptation&nominal
    checks['400_adaptation_seeds_disjoint_from_2400_nominal_seeds']=True
    stress={1_000_000+100_000*r+30000+j for r in range(10) for j in range(20)}
    small_exact={1_000_000+100_000*r+50000+j for r in range(10) for j in range(6)}
    assert not adaptation&(stress|small_exact)
    checks['adaptation_disjoint_from_historical_stress_and_small_exact_seeds']=True
    for task in ['mapping','mapping_coordinate','risks','exact','public_sensitivity','factorial']:
        evidence=json.loads((OUT/task/'execution.json').read_text())
        assert evidence['status']=='PASS',task
    completion=pd.read_csv(OUT/'mapping/completion_per_replicate.csv')
    keys=['replicate','sources','anchors','budget']
    a=completion[completion.method=='sequential'].set_index(keys)
    b=completion[completion.method=='feedback'].set_index(keys)
    assert np.max(np.abs(a.f1-b.f1))<1e-12
    checks['sequential_feedback_tie_all_270_replication_conditions']=True
    traces=pd.read_csv(OUT/'mapping/objective_traces.csv.gz')
    for _,g in traces.groupby(['replicate','sources','anchors','budget'],sort=False):
        assert np.min(np.diff(g.objective))>=-1e-7
    checks['accepted_objectives_never_decrease']=True
    schema=pd.read_csv(OUT/'mapping/schema_cases.csv')
    assert len(schema)==40 and (schema.accepted==schema.expected).all()
    rename=pd.read_csv(OUT/'mapping/common_permutation_control_per_scene.csv')
    assert len(rename)==800
    old=pd.read_csv(ROOT/'results/semantic_alignment/per_scene.csv')
    old=old[old.condition=='mapped'].sort_values(['replicate','scene'])
    assert np.max(np.abs(rename.sort_values(['replicate','scene']).f1.to_numpy()-old.f1.to_numpy()))<1e-12
    checks['800_shared_renaming_scores_and_40_schema_cases']=True
    counts=pd.read_csv(OUT/'factorial/pooled_pair_counts.csv')
    counts=counts[(counts.scorer=='joint')&(counts.decoder=='constrained')][['tp','fp','fn','true_pairs']].sum()
    assert tuple(counts.astype(int))==(30063,3589,13088,43151)
    checks['macro_micro_pair_denominator_trace']=counts.astype(int).to_dict()
    # Keep all original primary scientific figures byte-identical.
    baseline=json.loads((ROOT/'major_review_audit/baseline_manifest.json').read_text())
    protected=[x for x in baseline if x['path'].startswith('figures/') or x['path']=='code/experiment.py']
    for record in protected:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256'],record['path']
    checks['preserved_original_figure_assets_and_generator']=len(protected)
    # This check is available in the workspace copy; packaged projects may omit parent inputs.
    originals=json.loads((ROOT/'major_review_audit/protected_inputs.json').read_text(encoding='utf-8'))
    existing=[x for x in originals if Path(x['path']).is_file()]
    for record in existing:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256'],record['path']
    checks['original_workspace_inputs_unchanged']=len(existing)
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',checks=checks),indent=2))

if __name__=='__main__':main()
