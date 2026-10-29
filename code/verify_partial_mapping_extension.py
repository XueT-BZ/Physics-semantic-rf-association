"""Independently verify controlled and public missing-map predictions and constraints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment import simulate,Config,features
from partial_mapping_extension import ObservableBatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/partial_mapping_extension'

def independent_counts(reference,prediction):
    # Direct pair enumeration for small controlled scenes, separate from scorer.
    i,j=np.triu_indices(len(reference),1)
    t=reference[i]==reference[j];p=prediction[i]==prediction[j]
    tp=int(np.sum(t&p));fp=int(np.sum(~t&p));fn=int(np.sum(t&~p))
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn))

def main():
    checks={}
    assert set(ObservableBatch.__dataclass_fields__)=={'receiver','native','i','j','physical','time_z','freq_z','gate_k','kind'}
    checks['inference_interface_has_no_identity_or_hidden_mapping']='PASS'
    metrics=pd.read_csv(OUT/'controlled_per_scene.csv.gz').set_index(['replicate','budget','method','scene'])
    prediction=pd.read_csv(OUT/'controlled_predictions.csv.gz');cache={};n=0
    for key,g in prediction.groupby(['replicate','budget','method','scene']):
        rep,budget,method,index=key
        if (rep,index) not in cache:
            scene=simulate(1_000_000+100_000*rep+20_000+index,Config());cache[rep,index]=(scene,features(scene))
        scene,p=cache[rep,index];g=g.sort_values('observation')
        assert np.array_equal(g.reference.to_numpy(),scene.truth)
        z=g.predicted.to_numpy();value=independent_counts(scene.truth,z)
        assert abs(value['f1']-metrics.loc[key,'f1'])<1e-12,key
        assert not g.duplicated(['predicted','receiver']).any(),key
        assert not np.any((z[p.i]==z[p.j])&~p.gate()),key
        n+=1
    checks['controlled_prediction_groups_independently_recomputed']=n
    maps=json.loads((OUT/'controlled_maps.json').read_text());all_converged=True
    for item in maps:
        m=np.array(item['map']);hidden=np.array(item['hidden_map']);anchors={int(a):q for a,q in item['anchors'].items()}
        assert all(m[1,a]==q for a,q in anchors.items())
        assert np.array_equal(m[[0,2,3]],np.tile(np.arange(4),(3,1)))
        free=[a for a in range(4) if a not in anchors]
        assert np.mean(m[1,free]==hidden[1,free])==item['mapping_accuracy']
        all_converged &= item['converged']
    assert all_converged
    checks['anchors_and_withheld_mapping_metrics']='PASS'
    traces=pd.read_csv(OUT/'controlled_objective_trace.csv')
    for _,g in traces.groupby(['replicate','budget']):assert (np.diff(g.objective)>=-1e-8).all()
    pubtrace=pd.read_csv(OUT/'public_objective_trace.csv');assert (np.diff(pubtrace.objective)>=-1e-8).all()
    checks['objective_traces_monotone_and_runs_converged']='PASS'
    obs=pd.read_csv(ROOT/'results/loed/observations.csv.gz')
    refs=pd.read_csv(ROOT/'results/loed/evaluation_reference.csv.gz')
    data=obs.merge(refs,on='observation_id',validate='one_to_one')
    a=set(data[data.day=='2020-08-03'].reference_payload);b=set(data[data.day=='2020-09-02'].reference_payload)
    assert not a&b
    saved=pd.read_csv(OUT/'public_summary.csv').set_index('method')
    pred=pd.read_csv(OUT/'public_predictions.csv.gz')
    gate=json.loads((ROOT/'results/loed/fitted_parameters.json').read_text())['gate_s']
    for method,g in pred.groupby('method'):
        g=g.merge(data[['observation_id','reference_payload','corrected_time_s','window']],on='observation_id',validate='one_to_one')
        assert (g.reference==g.reference_payload).all()
        choose=lambda x:int(((x*(x-1))//2).sum())
        tp=choose(g.groupby(['reference','predicted']).size());fp=choose(g.groupby('predicted').size())-tp;fn=choose(g.groupby('reference').size())-tp
        assert all(value==saved.loc[method,key] for key,value in [('tp',tp),('fp',fp),('fn',fn)])
        assert not g.duplicated(['predicted','receiver']).any()
        pairs=g[g.duplicated('predicted',keep=False)].groupby('predicted')
        for _,pair in pairs:
            assert pair.window.nunique()==1
            assert pair.corrected_time_s.max()-pair.corrected_time_s.min()<=gate+1e-10
    checks['public_prediction_methods_recomputed']=len(saved)
    checks['public_reference_overlap_between_adaptation_and_test']=0
    checks['public_constraints_recomputed_from_timestamps']='PASS'
    checks['status']='PASS'
    target=ROOT/'audit/verification/partial_mapping.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
