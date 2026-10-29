"""Paired descriptor interventions for covariance, false alarms, NLOS, and fragments."""
from dataclasses import replace
from itertools import combinations
import json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import Scene,features,constrained_cluster,connected_cluster,scene_csv,UNIT_SCALE
from review_common import ROOT,OUT,parameters,test_scenes,score,metric,summary,environment

DEST=OUT/'risks'
CASES=['nominal','positive_corr_independent_declaration','positive_corr_matched',
       'negative_corr_independent_declaration','negative_corr_matched','nlos_excess_delay',
       'independent_false_alarms_matched_count','correlated_false_alarms','fragmented_detections']

def copy_scene(scene):
    return Scene({k:v.copy() for k,v in scene.obs.items()},scene.truth.copy(),scene.clutter.copy(),scene.positions.copy(),scene.seed)

def correlated(scene,rho,seed):
    result=copy_scene(scene);s=result.obs['receiver'];S=len(result.positions)
    generator=np.random.default_rng(scene.seed)
    generator.uniform(0,2*np.pi)
    old=generator.normal(0,.8e-6,S)
    covariance=(.8e-6)**2*((1-rho)*np.eye(S)+rho*np.ones((S,S)))
    vals,vecs=np.linalg.eigh(covariance)
    residual=vecs@np.diag(np.sqrt(np.maximum(vals,0)))@np.random.default_rng(seed).normal(size=S)
    result.obs['time_s']+=residual[s]-old[s]
    return result,covariance

def matched_features(scene,covariance):
    p=features(scene);o=scene.obs;s=o['receiver'];i=p.i;j=p.j
    variance=o['sigma_time_s'][i]**2+o['sigma_time_s'][j]**2+covariance[s[i],s[i]]+covariance[s[j],s[j]]-2*covariance[s[i],s[j]]
    u=np.sqrt(variance);dt=np.abs(o['time_s'][i]-o['time_s'][j])
    h=np.linalg.norm(scene.positions[s[i]]-scene.positions[s[j]],axis=1)/299792458.
    z=np.maximum(0,dt-h)/u
    designs=[]
    for original in [p.X,p.Xraw,p.Xunweighted]:
        x=original.copy();x[:,0]=np.log1p(dt/u);x[:,1]=np.log1p(z);designs.append(x)
    return replace(p,X=designs[0],Xraw=designs[1],Xunweighted=designs[2],time_z=z)

def false_alarms(scene,seed,shared=True):
    result=copy_scene(scene);rng=np.random.default_rng(seed);S=len(scene.positions)
    extra={k:[] for k in result.obs};labels=[]
    templates=rng.choice(np.flatnonzero(~scene.clutter),size=(3,S))
    for group in range(3):
        for receiver in range(S):
            index=int(templates[group,0 if shared else receiver]);original_receiver=int(scene.obs['receiver'][index])
            canonical=np.empty(4);canonical[(np.arange(4)+original_receiver)%4]=scene.obs['native_semantic'][index]
            f=scene.obs['frequency_value'][index]*UNIT_SCALE[scene.obs['frequency_unit'][index]]
            row={k:v[index].copy() if isinstance(v[index],np.ndarray) else v[index] for k,v in scene.obs.items()}
            row['receiver']=receiver
            row['time_s']+=rng.normal(0,.2e-6)
            row['frequency_value']=(f+rng.normal(0,120))/UNIT_SCALE[row['frequency_unit']]
            row['sigma_time_s']=(.6+.4*receiver)*1e-6
            row['sigma_frequency_hz']=(.12+.08*receiver)*1e3
            row['sigma_bandwidth_hz']=(1.5+.3*receiver)*1e3
            row['sigma_duration_s']=(1+.3*receiver)*1e-6
            row['native_semantic']=canonical[(np.arange(4)+receiver)%4]
            for key,value in row.items():extra[key].append(value)
            labels.append(int(scene.truth.max())+1+len(labels))
    result.obs={k:np.concatenate([result.obs[k],np.asarray(extra[k])]) for k in result.obs}
    result.truth=np.concatenate([result.truth,np.asarray(labels)])
    result.clutter=np.concatenate([result.clutter,np.ones(len(labels),bool)])
    return result

def fragments(scene,seed):
    result=copy_scene(scene);rng=np.random.default_rng(seed)
    selected=np.flatnonzero((~scene.clutter)&(rng.random(len(scene.truth))<.2))
    result.obs={k:np.concatenate([v,v[selected].copy()]) for k,v in result.obs.items()}
    if len(selected):
        result.obs['time_s'][-len(selected):]+=rng.normal(0,.15e-6,len(selected))
        units=result.obs['frequency_unit'][-len(selected):]
        result.obs['frequency_value'][-len(selected):]+=rng.normal(0,40,len(selected))/UNIT_SCALE[units]
    result.truth=np.concatenate([result.truth,scene.truth[selected]])
    result.clutter=np.concatenate([result.clutter,np.zeros(len(selected),bool)])
    return result

def recall_ceiling(scene):
    numerator=0;denominator=0
    for event in np.unique(scene.truth):
        receiver=scene.obs['receiver'][scene.truth==event]
        _,counts=np.unique(receiver,return_counts=True)
        numerator+=sum(min(int(a),int(b)) for a,b in combinations(counts,2))
        denominator+=len(receiver)*(len(receiver)-1)//2
    return numerator/max(1,denominator)

def main():
    DEST.mkdir(parents=True,exist_ok=True);environment('risks');started=time.perf_counter()
    protocol=dict(existing_test_scenes=200,replications=10,scenes_per_replication=20,cases=CASES,
                  covariance_positive_rho=.8,covariance_negative_rho=-.25,clock_marginal_sd_us=.8,
                  nlos_probability=.3,nlos_mean_us=8.,shared_false_alarm_templates=3,
                  matched_false_alarm_control='Same 12 added reports; independently sampled templates per receiver. Correlated intervention shares each template across receivers.',
                  fragmentation_probability=.2,seed_base=86_000_000,scorers=['physics','joint'],
                  decoders=['components','assignment','constrained'],refit=False,
                  correlated_false_alarm_truth='Each inserted false alarm retains its own singleton reference; no common physical nuisance event is claimed.',
                  fragmentation_truth='Inserted fragments retain the same true event and receiver; this intentionally breaks the consolidated-input assumption.')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];snapshots=[]
    for rep in range(10):
        params=parameters(rep);saved={};boundaries={name:[0] for name in CASES}
        predictions={}
        for index,base in enumerate(test_scenes(rep,20)):
            seed=86_000_000+rep*100_000+index
            positive,cov_pos=correlated(base,.8,seed)
            negative,cov_neg=correlated(base,-.25,seed)
            nlos=copy_scene(base);rng=np.random.default_rng(seed)
            nlos.obs['time_s']+=(rng.random(len(base.truth))<.3)*rng.exponential(8e-6,len(base.truth))
            scenes=dict(nominal=base,positive_corr_independent_declaration=positive,positive_corr_matched=positive,
                negative_corr_independent_declaration=negative,negative_corr_matched=negative,
                nlos_excess_delay=nlos,independent_false_alarms_matched_count=false_alarms(base,seed,False),
                correlated_false_alarms=false_alarms(base,seed,True),fragmented_detections=fragments(base,seed))
            for name,scene in scenes.items():
                p=matched_features(scene,cov_pos) if name=='positive_corr_matched' else matched_features(scene,cov_neg) if name=='negative_corr_matched' else features(scene)
                boundaries[name].append(boundaries[name][-1]+len(scene.truth))
                ceiling=recall_ceiling(scene)
                if rep==index==0:
                    snapshots.append(scene_csv(scene,rep,name,index))
                for scorer in ['physics','joint']:
                    prob=score(p,params,scorer)
                    for decoder in ['components','assignment','constrained']:
                        z=constrained_cluster(scene,p,prob) if decoder=='constrained' else connected_cluster(scene,p,prob,assignment=decoder=='assignment')
                        m=metric(scene,p,z)
                        if decoder=='constrained':
                            assert m['receiver_conflicts']==m['gate_violations']==0
                            assert m['recall']<=ceiling+1e-12
                        rows.append(dict(replicate=rep,scene=index,condition=name,scorer=scorer,decoder=decoder,
                                         observations=len(scene.truth),exclusivity_recall_ceiling=ceiling,**m))
                        predictions.setdefault(name+'_'+scorer+'_'+decoder,[]).append(z)
        saved={key:np.concatenate(value) for key,value in predictions.items()}
        saved.update({name+'_boundaries':np.asarray(value) for name,value in boundaries.items()})
        np.savez_compressed(DEST/f'predictions_r{rep:02d}.npz',**saved)
        print(f'Observation-risk audit {rep+1}/10; {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'per_scene.csv.gz',index=False)
    columns=['f1','ari','b3f1','exact','receiver_violation','gate_violation','clutter_attachment','true_pair_gate_recall','exclusivity_recall_ceiling']
    rep=frame.groupby(['replicate','condition','scorer','decoder'],as_index=False)[columns].mean()
    rep.to_csv(DEST/'per_replicate.csv',index=False)
    summary(rep,['condition','scorer','decoder'],columns).to_csv(DEST/'summary.csv',index=False)
    pd.concat(snapshots,ignore_index=True).to_csv(DEST/'example_observations_r00.csv.gz',index=False)
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-started,rows=len(frame)),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
