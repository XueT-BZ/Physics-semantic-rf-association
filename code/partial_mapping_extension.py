"""Optional incomplete-registry probe; preserve the original scorer and decoder.

Inference accepts observable-only batches. Hidden maps and event references live
in benchmark adapters/evaluation, never in fit_mapping(). No classifier is refit.
"""
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from experiment import Config,Scene,Pairs,simulate,features,constrained_cluster,cluster_metrics,summarize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/partial_mapping_extension'
EPS=1e-12

@dataclass(frozen=True)
class ObservableBatch:
    receiver:np.ndarray
    native:np.ndarray
    i:np.ndarray
    j:np.ndarray
    physical:np.ndarray
    time_z:np.ndarray
    freq_z:np.ndarray
    gate_k:float
    kind:str

def common(batch,maps):
    q=np.empty_like(batch.native)
    for s in np.unique(batch.receiver):
        rows=np.flatnonzero(batch.receiver==s)
        q[np.ix_(rows,maps[s])]=batch.native[rows]
    return q

def probabilities(batch,maps,params):
    q=common(batch,maps)
    if batch.kind=='controlled':
        k=q.shape[1]
        r=1+np.sum(q*np.log(np.maximum(q,EPS)),axis=1)/np.log(k)
        sem=r[batch.i]*r[batch.j]*np.log(np.maximum(k*np.sum(q[batch.i]*q[batch.j],axis=1),EPS))
    else:
        # Original LoED model uses decoded SF mismatch, not invented posteriors.
        category=np.argmax(q,axis=1)
        sem=(category[batch.i]!=category[batch.j]).astype(float)
    x=np.column_stack([batch.physical,sem])[:,params['columns']]
    value=((x-np.array(params['mean']))/np.array(params['scale']))@np.array(params['coefficient'])+params['intercept']
    return expit(value/params['temperature'])

def weights(batch,maps,params):
    # Exactly the numerical signed weights used by the released decoder.
    return logit(np.clip(probabilities(batch,maps,params),EPS,1-EPS))

def decode(batch,maps,params):
    n=len(batch.receiver);zeros=np.zeros(len(batch.i))
    dummy=Scene({'receiver':batch.receiver},np.zeros(n,int),np.zeros(n,bool),np.empty((0,2)),0)
    p=Pairs(batch.physical,batch.physical,batch.physical,zeros,batch.i,batch.j,
            batch.time_z,batch.freq_z,np.zeros(len(zeros),bool))
    return constrained_cluster(dummy,p,probabilities(batch,maps,params),k=batch.gate_k)

def objective(batches,partitions,maps,params):
    return float(sum(weights(b,maps,params)[z[b.i]==z[b.j]].sum() for b,z in zip(batches,partitions)))

def candidates(anchors,k):
    rows=[a for a in range(k) if a not in anchors]
    values=sorted(set(range(k))-set(anchors.values()))
    assert len(rows)==len(values)
    result=[]
    for order in permutations(values):
        row=np.empty(k,int)
        for a,q in anchors.items():row[a]=q
        row[rows]=order;result.append(row)
    return result

def map_step(batches,partitions,maps,params,target,choices):
    best=maps.copy();score=objective(batches,partitions,best,params)
    for row in choices:
        proposal=maps.copy();proposal[target]=row
        value=objective(batches,partitions,proposal,params)
        if value>score+1e-10:best=proposal;score=value
    return best,score

def fit_mapping(batches,joint,physical,known_maps,target,anchors,cap=10):
    """One partially unknown source; exact finite map step, original group step."""
    choices=candidates(anchors,known_maps.shape[1])
    initial=known_maps.copy();initial[target]=choices[0]
    groups=[decode(b,initial,physical) for b in batches]
    first_score=objective(batches,groups,initial,joint)
    learned,score=map_step(batches,groups,initial,joint,target,choices)
    sequential=learned.copy()
    trace=[dict(iteration=-1,objective=first_score,map_changes=0,group_changes=0),
           dict(iteration=0,objective=score,map_changes=int(np.sum(initial!=learned)),group_changes=0)]
    converged=False
    for iteration in range(1,cap+1):
        changes=0
        for t,(b,old) in enumerate(zip(batches,groups)):
            proposed=decode(b,learned,joint)
            if objective([b],[proposed],learned,joint)>objective([b],[old],learned,joint)+1e-10:
                groups[t]=proposed;changes+=1
        updated,score=map_step(batches,groups,learned,joint,target,choices)
        map_changes=int(np.sum(updated!=learned));learned=updated
        assert score>=trace[-1]['objective']-1e-8
        trace.append(dict(iteration=iteration,objective=score,map_changes=map_changes,group_changes=changes))
        if changes==0 and map_changes==0:converged=True;break
    return dict(completion=initial,sequential=sequential,feedback=learned,
                partitions=groups,trace=trace,converged=converged,candidate_count=len(choices))

def controlled_adapter(scene,hidden):
    old=scene.obs['native_semantic'];recv=scene.obs['receiver'];canonical=np.empty_like(old)
    for row,s in enumerate(recv):canonical[row,(np.arange(4)+s)%4]=old[row]
    native=np.empty_like(old)
    for s in np.unique(recv):
        selected=np.flatnonzero(recv==s);native[selected]=canonical[selected][:,hidden[s]]
    p=features(scene)
    b=ObservableBatch(recv.copy(),native,p.i,p.j,p.X[:,:5].copy(),p.time_z,p.freq_z,3.5,'controlled')
    return b,p

def accuracy(estimated,hidden,target,anchors):
    free=[a for a in range(hidden.shape[1]) if a not in anchors]
    return float(np.mean(estimated[target,free]==hidden[target,free]))

def constraints(batch,z):
    conflicts=sum(len(np.unique(batch.receiver[z==q]))!=int(np.sum(z==q)) for q in np.unique(z))
    same=z[batch.i]==z[batch.j]
    violations=np.sum(same&((batch.time_z>batch.gate_k)|(batch.freq_z>batch.gate_k)))
    return int(conflicts),int(violations)

def run_controlled():
    rows=[];mapping_rows=[];traces=[];predictions=[];oracle_matches=0
    original=pd.read_csv(ROOT/'results/semantic_alignment/per_scene.csv')
    original=original[original.condition=='mapped'].set_index(['replicate','scene'])
    for rep in range(10):
        base=1_000_000+100_000*rep
        params=json.loads((ROOT/f'results/linear_models_r{rep:02d}.json').read_text())
        hidden=np.tile(np.arange(4),(4,1));hidden[1]=np.random.default_rng(9_100_000+rep).permutation(4)
        anchors={int(np.flatnonzero(hidden[1]==0)[0]):0};known=np.tile(np.arange(4),(4,1))
        adapt=[controlled_adapter(simulate(base+10_000+i,Config()),hidden)[0] for i in range(40)]
        scenes=[simulate(base+20_000+i,Config()) for i in range(80)]
        test=[controlled_adapter(s,hidden) for s in scenes]
        cache={}
        def evaluate(maps):
            nonlocal oracle_matches
            key=tuple(maps[1])
            if key not in cache:
                results=[]
                for index,(scene,(b,p)) in enumerate(zip(scenes,test)):
                    z=decode(b,maps,params['joint']);metric=cluster_metrics(scene,p,z)
                    conflict,violation=constraints(b,z)
                    assert conflict==violation==0
                    if np.array_equal(maps,hidden):
                        assert abs(metric['f1']-original.loc[(rep,index),'f1'])<1e-12
                        oracle_matches+=1
                    results.append((metric,z,conflict,violation))
                cache[key]=results
            return cache[key]
        for budget in [1,5,40]:
            answer=fit_mapping(adapt[:budget],params['joint'],params['physics'],known,1,anchors)
            maps={name:answer[name] for name in ['completion','sequential','feedback']};maps['known']=hidden
            for step in answer['trace']:traces.append(dict(replicate=rep,budget=budget,**step))
            for method,m in maps.items():
                acc=accuracy(m,hidden,1,anchors)
                mapping_rows.append(dict(replicate=rep,budget=budget,method=method,mapping_accuracy=acc,
                    map=m.tolist(),hidden_map=hidden.tolist(),anchors=anchors,converged=answer['converged']))
                for index,(metric,z,conflict,violation) in enumerate(evaluate(m)):
                    rows.append(dict(replicate=rep,budget=budget,method=method,scene=index,
                        mapping_accuracy=acc,receiver_conflicts=conflict,gate_violations=violation,**metric))
                    if budget in [1,40]:
                        b=test[index][0]
                        predictions.extend(dict(replicate=rep,budget=budget,method=method,scene=index,
                            observation=t,receiver=int(b.receiver[t]),reference=int(scenes[index].truth[t]),
                            predicted=int(z[t])) for t in range(len(z)))
        print('Controlled partial registry:',rep+1,'/10',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'controlled_per_scene.csv.gz',index=False)
    means=d.groupby(['replicate','budget','method'],as_index=False)[['f1','mapping_accuracy','receiver_conflicts','gate_violations']].mean()
    means.to_csv(OUT/'controlled_per_replicate.csv',index=False)
    summary=summarize(means,['budget','method'],['f1']).merge(
        means.groupby(['budget','method'],as_index=False).mapping_accuracy.mean(),on=['budget','method'])
    full=pd.DataFrame(mapping_rows).groupby(['budget','method']).mapping_accuracy.agg(
        fully_recovered_sources=lambda x:int(np.sum(np.asarray(x)==1)),
        unknown_entries_correct=lambda x:int(round(np.sum(x)*3)),
        unknown_entries_total=lambda x:int(len(x)*3)).reset_index()
    summary=summary.merge(full,on=['budget','method'])
    summary.to_csv(OUT/'controlled_summary.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'controlled_objective_trace.csv',index=False)
    (OUT/'controlled_maps.json').write_text(json.dumps(mapping_rows,indent=2),encoding='utf-8')
    pd.DataFrame(predictions).to_csv(OUT/'controlled_predictions.csv.gz',index=False)
    differences=[]
    for budget,g in means.groupby('budget'):
        wide=g.pivot(index='replicate',columns='method',values='f1')
        differences.extend(dict(replicate=rep,budget=budget,delta=float(wide.loc[rep,'feedback']-wide.loc[rep,'sequential'])) for rep in wide.index)
    dif=pd.DataFrame(differences)
    summarize(dif,['budget'],['delta']).to_csv(OUT/'controlled_feedback_difference.csv',index=False)
    assert oracle_matches==800,oracle_matches
    return dict(known_map_scene_f1_matches_original=oracle_matches,repetitions=10,test_scenes=800)

def public_batches(frame,hidden,gate):
    batches=[];locations=[]
    for _,g in frame.groupby('window',sort=True):
        g=g.copy();loc=g.index.to_numpy();recv=g.local_receiver.to_numpy();q=g.spreading_factor.to_numpy(dtype=int)-7
        native=np.empty((len(g),6))
        for s in np.unique(recv):
            selected=np.flatnonzero(recv==s)
            native[selected]=np.eye(6)[q[selected]][:,hidden[s]]
        i,j=np.triu_indices(len(g),1);keep=recv[i]!=recv[j];i=i[keep];j=j[keep]
        time_s=g.corrected_time_s.to_numpy();tz=np.abs(time_s[i]-time_s[j])/gate
        f=g.frequency.to_numpy();physical=np.column_stack([np.log1p(tz),(f[i]!=f[j]).astype(float)])
        batches.append(ObservableBatch(recv,native,i,j,physical,tz,np.zeros(len(i)),1.,'public'))
        locations.append(loc)
    return batches,locations

def pair_counts(reference,predicted):
    frame=pd.DataFrame(dict(r=reference,p=predicted))
    choose=lambda sizes:int(((sizes*(sizes-1))//2).sum())
    tp=choose(frame.groupby(['r','p']).size());fp=choose(frame.groupby('p').size())-tp;fn=choose(frame.groupby('r').size())-tp
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn))

def run_public():
    observations=pd.read_csv(ROOT/'results/loed/observations.csv.gz')
    reference=pd.read_csv(ROOT/'results/loed/evaluation_reference.csv.gz')
    params=json.loads((ROOT/'results/loed/fitted_parameters.json').read_text())
    adapt=observations[observations.day=='2020-08-03'].copy().reset_index(drop=True)
    test=observations[observations.day=='2020-09-02'].copy().reset_index(drop=True)
    active=sorted(set(adapt.receiver)|set(test.receiver));assert len(active)==2
    lookup=dict(zip(active,range(2)))
    for frame in [adapt,test]:frame['local_receiver']=frame.receiver.map(lookup)
    hidden=np.tile(np.arange(6),(2,1));hidden[1,[0,5]]=hidden[1,[5,0]]
    anchors={i:i for i in [1,2,3,4]};known=np.tile(np.arange(6),(2,1))
    a,_=public_batches(adapt,hidden,params['gate_s']);b,locations=public_batches(test,hidden,params['gate_s'])
    answer=fit_mapping(a,params['models']['metadata'],params['models']['timing'],known,1,anchors)
    refs=test[['observation_id']].merge(reference,on='observation_id',validate='one_to_one').reference_payload.to_numpy()
    rows=[];predictions=[]
    for method,maps in [('completion',answer['completion']),('sequential',answer['sequential']),
                        ('feedback',answer['feedback']),('known',hidden)]:
        labels=np.empty(len(test),int);offset=0;conflicts=0;violations=0
        for batch,loc in zip(b,locations):
            z=decode(batch,maps,params['models']['metadata']);labels[loc]=z+offset;offset+=int(z.max())+1
            c,v=constraints(batch,z);conflicts+=c;violations+=v
        metric=pair_counts(refs,labels)
        if method=='known':
            original=pd.read_csv(ROOT/'results/loed/per_day.csv')
            wanted=original[(original.day=='2020-09-02')&(original.scorer=='metadata')&(original.decoder=='constrained')].iloc[0]
            assert all(metric[key]==wanted[key] for key in ['tp','fp','fn'])
        assert conflicts==violations==0
        rows.append(dict(method=method,mapping_accuracy=accuracy(maps,hidden,1,anchors),adaptation_reports=len(adapt),
            test_reports=len(test),receiver_conflicts=conflicts,gate_violations=violations,**metric))
        predictions.extend(dict(observation_id=id,method=method,receiver=int(recv),reference=r,predicted=int(z))
            for id,recv,r,z in zip(test.observation_id,test.receiver,refs,labels))
    pd.DataFrame(rows).to_csv(OUT/'public_summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'public_predictions.csv.gz',index=False)
    pd.DataFrame(answer['trace']).to_csv(OUT/'public_objective_trace.csv',index=False)
    metadata=dict(active_original_receiver_ids=active,unknown_common_classes=['SF7','SF12'],
        anchors=anchors,hidden_map=hidden.tolist(),sequential_map=answer['sequential'].tolist(),feedback_map=answer['feedback'].tolist(),
        adaptation_day='2020-08-03',test_day='2020-09-02',converged=answer['converged'],candidate_count=answer['candidate_count'],
        observed_adaptation_class_counts=adapt.groupby(['local_receiver','spreading_factor']).size().to_dict())
    metadata['observed_adaptation_class_counts']={str(k):int(v) for k,v in metadata['observed_adaptation_class_counts'].items()}
    # Convert numpy source IDs for a portable JSON record.
    metadata['active_original_receiver_ids']=[int(v) for v in active]
    (OUT/'public_maps.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return dict(adaptation_reports=len(adapt),test_reports=len(test),known_map_counts_match_original=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    report=dict(controlled=run_controlled(),public=run_public())
    report.update(status='PASS',seconds=time.perf_counter()-start,
        original_generator_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'PARTIAL_MAPPING_PROTOCOL.txt').read_bytes()).hexdigest(),
        no_refitted_scorers=True,original_entropy_weighted_score_preserved=True)
    (OUT/'execution_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
