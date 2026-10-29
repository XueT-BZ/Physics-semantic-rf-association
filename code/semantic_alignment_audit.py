"""Audit representation and known-vocabulary alignment on existing RF scenes.

No scenes are generated with new distributions and no model is refitted. The
800 released test scenes and frozen joint-linear scorer are reused. Explicit
per-receiver mapping registries are tested, not learned ontology matching.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiment import (Config, Scene, Pairs, simulate, features, UNIT_SCALE,
                        constrained_cluster, cluster_metrics, summarize)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/semantic_alignment'
KINDS=['mapped','native_order_bypassed','class_evidence_neutral']


def canonical_probabilities(scene):
    native=scene.obs['native_semantic']; pi=np.empty_like(native)
    for n,s in enumerate(scene.obs['receiver']):
        pi[n,(np.arange(4)+s)%4]=native[n]
    return pi


def reencode_and_restore(scene, pi):
    """Change receiver-specific representations with matching declared maps."""
    obs={k:v.copy() for k,v in scene.obs.items()}
    old_units=obs['frequency_unit'].copy()
    hz=obs['frequency_value']*UNIT_SCALE[old_units]
    obs['frequency_unit']=(old_units+1+obs['receiver'])%3
    obs['frequency_value']=hz/UNIT_SCALE[obs['frequency_unit']]
    rng=np.random.default_rng(scene.seed+8_000_000)
    recovered=np.empty_like(pi)
    for s in np.unique(obs['receiver']):
        mask=obs['receiver']==s
        order=rng.permutation(4)  # native column -> common class index
        encoded=pi[mask][:,order]
        restored=np.empty_like(encoded);restored[:,order]=encoded
        recovered[mask]=restored
        # Adapt the common vector back to the original feature API's declared
        # cyclic native convention. No identities or labels enter this adapter.
        obs['native_semantic'][mask]=restored[:,(np.arange(4)+s)%4]
    assert np.array_equal(recovered,pi)
    return Scene(obs,scene.truth.copy(),scene.clutter.copy(),scene.positions.copy(),scene.seed)


def change_evidence(pairs, probability):
    q=np.clip(probability,1e-12,1.)
    concentration=1+np.sum(q*np.log(q),axis=1)/np.log(4)
    evidence=np.log(np.maximum(4*np.sum(q[pairs.i]*q[pairs.j],axis=1),1e-12))
    sem=concentration[pairs.i]*concentration[pairs.j]*evidence
    x=pairs.X.copy();x[:,-1]=sem
    xu=pairs.Xunweighted.copy();xu[:,-1]=evidence
    raw=pairs.Xraw.copy();raw[:,-1]=sem
    return Pairs(x,raw,xu,pairs.y.copy(),pairs.i.copy(),pairs.j.copy(),
                 pairs.time_z.copy(),pairs.freq_z.copy(),pairs.unseen.copy())


def predict(pairs, params):
    x=pairs.X[:,params['columns']]
    z=((x-np.array(params['mean']))/np.array(params['scale']))@np.array(params['coefficient'])+params['intercept']
    return expit(z/params['temperature'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];max_feature=0.;max_probability=0.;matched=0
    for rep in range(10):
        base=1_000_000+100_000*rep
        params=json.loads((ROOT/f'results/linear_models_r{rep:02d}.json').read_text())['joint']
        old=np.load(ROOT/f'results/factorial/predictions_r{rep:02d}.npz')
        saved={k:[] for k in KINDS};saved['representation_reencoded']=[]
        boundaries=[0]
        for index in range(80):
            scene=simulate(base+20_000+index,Config());pairs=features(scene)
            pi=canonical_probabilities(scene)
            assert np.allclose(change_evidence(pairs,pi).X,pairs.X,atol=1e-12)
            restored=features(reencode_and_restore(scene,pi))
            max_feature=max(max_feature,float(np.max(np.abs(restored.X-pairs.X))))
            assert np.allclose(restored.X,pairs.X,atol=1e-8,rtol=1e-10)
            assert np.array_equal(restored.gate(),pairs.gate())
            probability=predict(pairs,params);restored_probability=predict(restored,params)
            max_probability=max(max_probability,float(np.max(np.abs(probability-restored_probability))))
            a,b=old['pair_boundaries'][index:index+2]
            assert np.allclose(probability,old['score_joint'][a:b],atol=1e-10)
            z=constrained_cluster(scene,pairs,probability)
            zr=constrained_cluster(scene,restored,restored_probability)
            assert np.array_equal(z,zr)
            a,b=old['observation_boundaries'][index:index+2]
            assert np.array_equal(z,old['joint_constrained'][a:b])
            saved['representation_reencoded'].append(zr);matched+=1
            variants={'mapped':pairs,
                      'native_order_bypassed':change_evidence(pairs,scene.obs['native_semantic']),
                      'class_evidence_neutral':change_evidence(pairs,np.full_like(pi,.25))}
            for kind,p in variants.items():
                assert np.array_equal(p.gate(),pairs.gate())
                label=z if kind=='mapped' else constrained_cluster(scene,p,predict(p,params))
                saved[kind].append(label)
                metrics=cluster_metrics(scene,p,label)
                rows.append(dict(replicate=rep,scene=index,condition=kind,**metrics))
            boundaries.append(boundaries[-1]+len(scene.truth))
        np.savez_compressed(OUT/f'predictions_r{rep:02d}.npz',
                            **{k:np.concatenate(v) for k,v in saved.items()},
                            observation_boundaries=np.array(boundaries))
        print(f'Representation/semantic alignment audit: replication {rep+1}/10',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_scene.csv',index=False)
    columns=['f1','precision','recall','ari','exact','receiver_violation','gate_violation']
    means=frame.groupby(['replicate','condition'],as_index=False)[columns].mean()
    means.to_csv(OUT/'per_replicate.csv',index=False)
    summary=summarize(means,['condition'],columns)
    summary.to_csv(OUT/'summary.csv',index=False)
    wide=means.pivot(index='replicate',columns='condition',values='f1')
    differences=[]
    for condition in KINDS[1:]:
        for rep in range(10):
            differences.append(dict(replicate=rep,comparison=condition,
                                    delta=float(wide.loc[rep,'mapped']-wide.loc[rep,condition])))
    delta=pd.DataFrame(differences);delta.to_csv(OUT/'paired_per_replicate.csv',index=False)
    summarize(delta,['comparison'],['delta']).to_csv(OUT/'paired_summary.csv',index=False)
    lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule',
           r'Input treatment & Event-pair F1 [95\% CI] & Receiver conflict & Gate violation \\\midrule']
    labels={'mapped':'Declared class mapping applied',
            'native_order_bypassed':'Native columns treated as common',
            'class_evidence_neutral':'Class evidence set to uniform'}
    for kind in KINDS:
        row=summary[summary.condition==kind].iloc[0]
        lines.append(f"{labels[kind]} & {row.f1:.4f} [{row.f1_lo:.4f}, {row.f1_hi:.4f}] & {100*row.receiver_violation:.2f}\\% & {100*row.gate_violation:.2f}\\% "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabular*}')
    (ROOT/'tables/semantic_alignment.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=dict(status='PASS',existing_test_scenes=800,performance_conditions=3,
        performance_scene_conditions=len(frame),reencoded_scenes_with_identical_partition=matched,
        max_reencoded_feature_difference=max_feature,max_reencoded_probability_difference=max_probability,
        receiver_specific_declared_class_permutations=True,unknown_mapping_learned=False,
        scorer_and_gate_frozen=True,new_scene_distributions_added=False,
        original_experiment_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Known representation mappings and their association consequences; not real protocol recognition or unknown ontology matching.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(summary[['condition','f1','f1_lo','f1_hi']].to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
