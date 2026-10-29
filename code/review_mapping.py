"""Mapping corruption and multi-source map completion with frozen RF scorers."""
from dataclasses import replace
from itertools import product,combinations
import json,time,shutil,math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
from experiment import Config,simulate,features,scene_csv,cluster_metrics
from partial_mapping_extension import ObservableBatch,controlled_adapter,candidates,decode,weights
from review_common import ROOT,OUT,parameters,test_scenes,metric,summary,environment,save_csv

DEST=OUT/'mapping'

def validate(batch,maps):
    k=maps.shape[1]
    if batch.native.ndim!=2 or batch.native.shape[1]!=k:
        raise ValueError('Unsupported class vocabulary size')
    if not np.all(np.isfinite(batch.native)) or np.any(batch.native<0):
        raise ValueError('Invalid class probabilities')
    if not np.allclose(batch.native.sum(axis=1),1):
        raise ValueError('Class probabilities must sum to one')
    if any(not np.array_equal(np.sort(row),np.arange(k)) for row in maps):
        raise ValueError('This registry requires a bijective class map')

def pool(batches,groups):
    receivers=[];native=[];ii=[];jj=[];physical=[];tz=[];fz=[];offset=0
    for b,z in zip(batches,groups):
        selected=z[b.i]==z[b.j]
        receivers.append(b.receiver);native.append(b.native)
        ii.append(b.i[selected]+offset);jj.append(b.j[selected]+offset)
        physical.append(b.physical[selected]);tz.append(b.time_z[selected]);fz.append(b.freq_z[selected])
        offset+=len(b.receiver)
    return ObservableBatch(np.concatenate(receivers),np.concatenate(native),np.concatenate(ii),np.concatenate(jj),
                           np.concatenate(physical),np.concatenate(tz),np.concatenate(fz),3.5,'controlled')

def exact_map_step(batch,maps,choices,params,*,max_joint_states=1_000_000):
    """Exact conditional map block using pair tables and relative permutations."""
    S=len(maps);K=maps.shape[1]
    domains=[choices[s] if s in choices else [maps[s]] for s in range(S)]
    sizes=tuple(len(d) for d in domains)
    states=math.prod(sizes)
    if states>max_joint_states:
        raise ValueError(f'Conditional map block has {states:,} states; configured limit is {max_joint_states:,}. Use an approximate map search or a smaller block.')
    total=np.zeros(sizes)
    recv_i=batch.receiver[batch.i];recv_j=batch.receiver[batch.j]
    for u,v in combinations(range(S),2):
        use=((recv_i==u)&(recv_j==v))|((recv_i==v)&(recv_j==u))
        ii=np.where(recv_i[use]==u,batch.i[use],batch.j[use])
        jj=np.where(recv_i[use]==u,batch.j[use],batch.i[use])
        U=batch.native[ii];V=batch.native[jj]
        rU=1+np.sum(U*np.log(np.maximum(U,1e-12)),axis=1)/np.log(K)
        rV=1+np.sum(V*np.log(np.maximum(V,1e-12)),axis=1)/np.log(K)
        cache={};table=np.empty((sizes[u],sizes[v]))
        for a,left in enumerate(domains[u]):
            for b,right in enumerate(domains[v]):
                relative=tuple(np.argsort(right)[left])
                if relative not in cache:
                    sem=rU*rV*np.log(np.maximum(K*np.sum(U*V[:,relative],axis=1),1e-12))
                    X=np.column_stack([batch.physical[use],sem])[:,params['columns']]
                    z=(((X-np.asarray(params['mean']))/np.asarray(params['scale']))@np.asarray(params['coefficient'])+params['intercept'])/params['temperature']
                    cache[relative]=float(logit(np.clip(expit(z),1e-12,1-1e-12)).sum())
                table[a,b]=cache[relative]
        shape=[1]*S;shape[u]=sizes[u];shape[v]=sizes[v]
        total+=table.reshape(shape)
    state=np.unravel_index(np.argmax(total),sizes)
    proposal=np.asarray([domains[s][state[s]] for s in range(S)])
    old_value=float(weights(batch,maps,params).sum())
    value=float(weights(batch,proposal,params).sum())
    assert abs(value-float(total[state]))<1e-6
    if value>old_value+1e-10:return proposal,value,states
    return maps,old_value,states

def fit_multiple(batches,joint,physical,known,anchor_maps,cap=10,strategy='exact'):
    """Observable-only fitting. The default enumerates the complete map block."""
    choices={s:candidates(anchor_maps[s],known.shape[1]) for s in sorted(anchor_maps)}
    maps=known.copy()
    for s in choices:maps[s]=choices[s][0]
    for b in batches:validate(b,maps)
    initial=maps.copy()
    groups=[decode(b,maps,physical) for b in batches]
    traces=[]
    def coordinate(groups,current,stage):
        selected=pool(batches,groups)
        value=float(weights(selected,current,joint).sum())
        traces.append(dict(stage=stage,cycle=-1,objective=value,map_changes=0))
        if strategy=='exact':
            updated,value,count=exact_map_step(selected,current,choices,joint)
            traces.append(dict(stage=stage,cycle=0,objective=value,map_changes=int(np.sum(updated!=current)),joint_candidates=count))
            return updated,value,True
        converged=False
        for cycle in range(cap):
            count=0
            for source,options in choices.items():
                best=current;best_value=value
                for option in options:
                    trial=current.copy();trial[source]=option
                    candidate=float(weights(selected,trial,joint).sum())
                    if candidate>best_value+1e-10:best=trial;best_value=candidate
                count+=int(np.sum(best!=current));current=best;value=best_value
            traces.append(dict(stage=stage,cycle=cycle,objective=value,map_changes=count))
            if count==0:converged=True;break
        return current,value,converged
    maps,value,seq_converged=coordinate(groups,maps,'sequential')
    sequential=maps.copy();feedback_converged=False
    for outer in range(cap):
        changed=0
        for i,b in enumerate(batches):
            proposed=decode(b,maps,joint)
            old=groups[i]
            old_score=float(weights(b,maps,joint)[old[b.i]==old[b.j]].sum())
            new_score=float(weights(b,maps,joint)[proposed[b.i]==proposed[b.j]].sum())
            if new_score>old_score+1e-10:groups[i]=proposed;changed+=1
        old_maps=maps.copy()
        maps,new_value,inner_converged=coordinate(groups,maps,f'feedback_{outer}')
        assert new_value>=value-1e-7
        value=new_value
        if changed==0 and np.array_equal(old_maps,maps) and inner_converged:
            feedback_converged=True;break
    return dict(unresolved=initial,sequential=sequential,feedback=maps,trace=traces,
                sequential_converged=seq_converged,feedback_converged=feedback_converged)

def withheld_accuracy(maps,hidden,anchors):
    entries=[(s,a) for s in anchors for a in range(hidden.shape[1]) if a not in anchors[s]]
    correct=sum(maps[s,a]==hidden[s,a] for s,a in entries)
    return correct/max(1,len(entries)),int(correct),len(entries)

def corruption(hidden,budget,rng):
    allocations=[row for row in product([0,2,3,4],repeat=4) if sum(row)==budget]
    sizes=allocations[int(rng.integers(len(allocations)))];result=hidden.copy()
    for s,n in enumerate(sizes):
        if n:
            rows=rng.choice(4,n,replace=False)
            result[s,rows]=result[s,np.roll(rows,1)]
    assert np.sum(result!=hidden)==budget
    return result

def main(strategy='exact'):
    global DEST
    DEST=OUT/('mapping' if strategy=='exact' else 'mapping_coordinate')
    DEST.mkdir(parents=True,exist_ok=True);environment('mapping');start=time.perf_counter()
    coordinate_copy=OUT/'mapping_coordinate'
    if strategy=='exact' and (DEST/'execution.json').exists() and not coordinate_copy.exists():
        shutil.copytree(DEST,coordinate_copy)
    protocol=dict(replications=10,test_scenes_per_replication=80,adaptation_budgets=[1,5,40],
                  unknown_receivers=[1,2,3],anchors_per_unknown_source=[0,1,2],
                  wrong_entries=[0,2,4,8,12,16],mapping_interventions_per_replication=3,
                  adaptation_seed_base=80_000_000,adaptation_scenes_are_disjoint_from_historical_training_and_temperature_calibration=True,
                  original_test_scenes_reused=True,post_review_secondary_audit=True,
                  fitted_classifiers_refit=False,maximum_map_cycles=10,maximum_feedback_sweeps=10,
                  known_reference_receiver=0,map_update=('Exact conditional joint enumeration using relative-permutation pair tables' if strategy=='exact' else 'Per-receiver coordinate enumeration'),
                  maximum_joint_map_candidates=24**3,coordinate_update_retained_under='mapping_coordinate',
                  unsupported_vocabularies='reject; no score is reported')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    wrong_rows=[];complete_rows=[];map_records=[];traces=[];adapt_frames=[];checks=[];rename_rows=[]
    legacy=pd.read_csv(ROOT/'results/semantic_alignment/per_scene.csv')
    legacy=legacy[legacy.condition=='mapped'].set_index(['replicate','scene'])
    for rep in range(10):
        params=parameters(rep);scenes=test_scenes(rep);pairs=[features(s) for s in scenes]
        true_cyclic=np.array([(np.arange(4)+s)%4 for s in range(4)])
        actual=[controlled_adapter(s,true_cyclic)[0] for s in scenes]
        saved={};boundaries=np.cumsum([0]+[len(s.truth) for s in scenes])
        renamed=(true_cyclic+1)%4
        renamed_labels=[]
        for i,(s,p,b) in enumerate(zip(scenes,pairs,actual)):
            z=decode(b,renamed,params['joint']);m=metric(s,p,z)
            original=decode(b,true_cyclic,params['joint'])
            assert np.array_equal(z[:,None]==z[None,:],original[:,None]==original[None,:])
            rename_rows.append(dict(replicate=rep,scene=i,wrong_fraction=1.,**m));renamed_labels.append(z)
        saved['common_class_renaming']=np.concatenate(renamed_labels)
        for budget in protocol['wrong_entries']:
            for intervention in range(3):
                maps=corruption(true_cyclic,budget,np.random.default_rng(82_000_000+rep*10_000+budget*10+intervention))
                labels=[]
                for i,(s,p,b) in enumerate(zip(scenes,pairs,actual)):
                    z=decode(b,maps,params['joint']);m=metric(s,p,z)
                    assert m['receiver_conflicts']==m['gate_violations']==0
                    if budget==0:assert abs(m['f1']-legacy.loc[(rep,i),'f1'])<1e-12
                    wrong_rows.append(dict(replicate=rep,budget=budget,wrong_fraction=budget/16,
                                           intervention=intervention,scene=i,**m));labels.append(z)
                saved[f'wrong_{budget}_{intervention}']=np.concatenate(labels)
        adaptation_scenes=[simulate(80_000_000+100_000*rep+j,Config()) for j in range(40)]
        adapt_frames.extend(scene_csv(s,rep,'review_adaptation',i) for i,s in enumerate(adaptation_scenes))
        for sources in [1,2,3]:
            hidden=np.tile(np.arange(4),(4,1));rng=np.random.default_rng(83_000_000+rep)
            for s in range(1,sources+1):hidden[s]=rng.permutation(4)
            adapt=[controlled_adapter(s,hidden)[0] for s in adaptation_scenes]
            test=[controlled_adapter(s,hidden)[0] for s in scenes]
            cache={}
            def evaluate(maps):
                key=tuple(maps.ravel())
                if key not in cache:
                    rows=[];zs=[]
                    for i,(s,p,b) in enumerate(zip(scenes,pairs,test)):
                        z=decode(b,maps,params['joint']);m=metric(s,p,z)
                        assert m['receiver_conflicts']==m['gate_violations']==0
                        if np.array_equal(maps,hidden):assert abs(m['f1']-legacy.loc[(rep,i),'f1'])<1e-12
                        rows.append(m);zs.append(z)
                    cache[key]=rows
                    saved[f'map_{sources}_{len(cache)-1}']=np.concatenate(zs)
                return cache[key]
            for anchor_count in [0,1,2]:
                anchors={s:{int(a):int(hidden[s,a]) for a in np.random.default_rng(84_000_000+100*rep+s).permutation(4)[:anchor_count]} for s in range(1,sources+1)}
                for budget in [1,5,40]:
                    result=fit_multiple(adapt[:budget],params['joint'],params['physics'],np.tile(np.arange(4),(4,1)),anchors,strategy=strategy)
                    for step in result['trace']:traces.append(dict(replicate=rep,sources=sources,anchors=anchor_count,budget=budget,**step))
                    maps_to_test={m:result[m] for m in ['unresolved','sequential','feedback']};maps_to_test['known']=hidden
                    for method,maps in maps_to_test.items():
                        acc,correct,total=withheld_accuracy(maps,hidden,anchors)
                        map_records.append(dict(replicate=rep,sources=sources,anchors=anchor_count,budget=budget,method=method,
                            mapping_accuracy=acc,correct=correct,total=total,map=maps.tolist(),hidden_evaluation_only=hidden.tolist(),
                            disclosed_entries=anchors,feedback_converged=result['feedback_converged'],sequential_converged=result['sequential_converged'],
                            prediction_key=f'map_{sources}_{list(cache).index(tuple(maps.ravel()))}' if tuple(maps.ravel()) in cache else f'map_{sources}_{len(cache)}'))
                        for i,m in enumerate(evaluate(maps)):
                            complete_rows.append(dict(replicate=rep,sources=sources,anchors=anchor_count,budget=budget,method=method,
                                mapping_accuracy=acc,correct=correct,total=total,scene=i,**m))
        probe=actual[0];maps=true_cyclic.copy()
        tests=[('valid_permutation',probe,maps,True),
               ('many_to_one',probe,np.array([[0,0,2,3],*maps[1:].tolist()]),False),
               ('coarser_three_class_vector',replace(probe,native=probe.native[:,:3]),maps,False),
               ('unknown_fifth_class',replace(probe,native=np.column_stack([.8*probe.native,np.full(len(probe.receiver),.2)])),maps,False)]
        for name,b,m,expected in tests:
            try:validate(b,m);accepted=True
            except ValueError:accepted=False
            assert accepted==expected
            checks.append(dict(replicate=rep,case=name,accepted=accepted,expected=expected))
        np.savez_compressed(DEST/f'predictions_r{rep:02d}.npz',observation_boundaries=boundaries,**saved)
        print(f'Mapping audit {rep+1}/10; {time.perf_counter()-start:.1f}s',flush=True)
    wrong=pd.DataFrame(wrong_rows);save_csv(wrong,DEST/'wrong_map_per_scene.csv.gz')
    wr=wrong.groupby(['replicate','budget','wrong_fraction'],as_index=False)[METRIC_COLUMNS].mean()
    save_csv(wr,DEST/'wrong_map_per_replicate.csv')
    save_csv(summary(wr,['budget','wrong_fraction'],METRIC_COLUMNS),DEST/'wrong_map_summary.csv')
    complete=pd.DataFrame(complete_rows);save_csv(complete,DEST/'completion_per_scene.csv.gz')
    cr=complete.groupby(['replicate','sources','anchors','budget','method'],as_index=False)[METRIC_COLUMNS+['mapping_accuracy']].mean()
    save_csv(cr,DEST/'completion_per_replicate.csv')
    aggregate=summary(cr,['sources','anchors','budget','method'],METRIC_COLUMNS)
    accuracy_frame=pd.DataFrame(map_records).groupby(['sources','anchors','budget','method'],as_index=False).agg(
        mapping_accuracy=('mapping_accuracy','mean'),mapping_accuracy_min=('mapping_accuracy','min'),
        mapping_accuracy_max=('mapping_accuracy','max'),withheld_correct=('correct','sum'),withheld_total=('total','sum'))
    save_csv(aggregate.merge(accuracy_frame,on=['sources','anchors','budget','method']),DEST/'completion_summary.csv')
    save_csv(pd.DataFrame(rename_rows),DEST/'common_permutation_control_per_scene.csv')
    save_csv(pd.DataFrame(traces),DEST/'objective_traces.csv.gz')
    save_csv(pd.DataFrame(checks),DEST/'schema_cases.csv')
    pd.concat(adapt_frames,ignore_index=True).to_csv(DEST/'independent_adaptation_observations.csv.gz',index=False)
    (DEST/'learned_maps.json').write_text(json.dumps(map_records,indent=2),encoding='utf-8')
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-start,wrong_map_rows=len(wrong),completion_rows=len(complete),schema_checks=len(checks)),indent=2),encoding='utf-8')

METRIC_COLUMNS=['f1','ari','b3f1','exact','receiver_violation','gate_violation']
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
