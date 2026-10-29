"""New-seed map-completion comparisons and bounded computational checks.

The spectral baseline adapts Pachauri, Kondor and Singh (NIPS 2013), Algorithm 1.
Our RF-specific preprocessing estimates each relative permutation from the SAME
fixed-group pair scores used by exact enumeration. The spectral stage receives
one winning permutation per observed receiver pair. It discards score margins.
This is an adapted baseline, not a reproduction of their image experiments.
"""
from dataclasses import dataclass
from itertools import permutations,combinations,product
from pathlib import Path
import argparse,json,time,math,platform,sys,tracemalloc
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from experiment import Config,simulate,features,scene_csv
from partial_mapping_extension import ObservableBatch,controlled_adapter
from review_mapping import pool
from review_common import metric,summary
from deep_scoring import load_scorers,ROOT

DEST=ROOT/'results/deep_revision/mapping'
MAX_STATES=1_000_000
METRICS=['f1','precision','recall','ari','b3f1','exact','receiver_violation','gate_violation']

@dataclass
class PairCache:
    permutations:np.ndarray
    relative_scores:dict
    tables:dict
    edge_counts:dict
    sizes:tuple

    @property
    def S(self):return len(self.sizes)
    @property
    def K(self):return self.permutations.shape[1]
    @property
    def states(self):return math.prod(self.sizes)

    def maps(self,state):return self.permutations[np.asarray(state)]

    def state(self,maps):
        lookup={tuple(p):i for i,p in enumerate(self.permutations)}
        return np.asarray([lookup[tuple(p)] for p in maps],int)

    def objective(self,state):
        return float(sum(t[state[u],state[v]] for (u,v),t in self.tables.items()))

    def score_array(self):
        if self.states>MAX_STATES:raise ValueError('Conditional state limit exceeded')
        total=np.zeros(self.sizes)
        for (u,v),table in self.tables.items():
            shape=[1]*self.S;shape[u]=self.sizes[u];shape[v]=self.sizes[v]
            total+=table.reshape(shape)
        return total

def relative_index(perms,left_count=None):
    lookup={tuple(p):i for i,p in enumerate(perms)}
    inv=np.argsort(perms,axis=1)
    left_count=len(perms) if left_count is None else left_count
    index=np.empty((left_count,len(perms)),int)
    for a,left in enumerate(perms[:left_count]):
        index[a]=[lookup[tuple(x[left])] for x in inv]
    return index

def cache_from_relative(K,S,scores,edge_counts):
    perms=np.asarray(list(permutations(range(K))),int)
    sizes=(1,)+(len(perms),)*(S-1)
    if math.prod(sizes)>MAX_STATES:raise ValueError('Conditional state limit exceeded')
    rel=relative_index(perms,len(perms) if S>2 else 1);tables={}
    for u,v in combinations(range(S),2):
        tables[u,v]=np.asarray(scores[u,v])[rel[:sizes[u],:sizes[v]]]
    return PairCache(perms,scores,tables,edge_counts,sizes)

def build_cache(batch,scorer,S=None):
    K=batch.native.shape[1];S=int(batch.receiver.max())+1 if S is None else S
    if math.factorial(K)**(S-1)>MAX_STATES:raise ValueError('Conditional state limit exceeded')
    perms=np.asarray(list(permutations(range(K))),int)
    ri=batch.receiver[batch.i];rj=batch.receiver[batch.j];scores={};counts={}
    for u,v in combinations(range(S),2):
        use=((ri==u)&(rj==v))|((ri==v)&(rj==u))
        ii=np.where(ri[use]==u,batch.i[use],batch.j[use])
        jj=np.where(ri[use]==u,batch.j[use],batch.i[use])
        U=batch.native[ii];V=batch.native[jj];E=len(ii);counts[u,v]=E
        if not E:scores[u,v]=np.zeros(len(perms));continue
        concentration=(1+np.sum(U*np.log(np.maximum(U,1e-12)),axis=1)/np.log(K))*(1+np.sum(V*np.log(np.maximum(V,1e-12)),axis=1)/np.log(K))
        compat=np.asarray([np.log(np.maximum(K*np.sum(U*V[:,p],axis=1),1e-12)) for p in perms])
        weights=scorer.weights_from(np.tile(batch.physical[use],(len(perms),1)),compat.ravel(),np.tile(concentration,len(perms)))
        scores[u,v]=weights.reshape(len(perms),E).sum(axis=1)
    return cache_from_relative(K,S,scores,counts)

def exact(cache,initial=None):
    initial=np.zeros(cache.S,int) if initial is None else np.asarray(initial).copy()
    total=cache.score_array();state=np.asarray(np.unravel_index(np.argmax(total),cache.sizes))
    if float(total[tuple(state)])<=cache.objective(initial)+1e-10:state=initial
    return state,dict(objective=cache.objective(state),cycles=1,converged=True)

def coordinate(cache,initial=None,cap=100):
    state=np.zeros(cache.S,int) if initial is None else np.asarray(initial).copy()
    converged=False
    for cycle in range(cap):
        changes=0
        for s in range(1,cache.S):
            values=np.zeros(cache.sizes[s])
            for (u,v),table in cache.tables.items():
                if u==s:values+=table[:,state[v]]
                elif v==s:values+=table[state[u],:]
            best=int(np.argmax(values))
            if values[best]>values[state[s]]+1e-10:state[s]=best;changes+=1
        if not changes:converged=True;break
    return state,dict(objective=cache.objective(state),cycles=cycle+1,converged=converged)

def reference_only(cache):
    state=np.zeros(cache.S,int)
    for s in range(1,cache.S):state[s]=int(np.argmax(cache.tables[0,s][0]))
    return state,dict(objective=cache.objective(state),cycles=1,converged=True)

def spectral(cache):
    S,K=cache.S,cache.K;T=np.eye(S*K)
    for (u,v),values in cache.relative_scores.items():
        if not cache.edge_counts[u,v]:continue
        rel=cache.permutations[int(np.argmax(values))]
        block=np.zeros((K,K));block[np.arange(K),rel]=1.
        T[u*K:(u+1)*K,v*K:(v+1)*K]=block
        T[v*K:(v+1)*K,u*K:(u+1)*K]=block.T
    eigenvalues,eigenvectors=np.linalg.eigh(T)
    U=np.sqrt(S)*eigenvectors[:,-K:]
    maps=np.tile(np.arange(K),(S,1))
    for s in range(1,S):
        block=U[s*K:(s+1)*K]@U[:K].T
        a,b=linear_sum_assignment(block,maximize=True);maps[s,a]=b
    state=cache.state(maps)
    return state,dict(objective=cache.objective(state),cycles=1,converged=True,
                      spectral_eigenvalues=eigenvalues[-K:].tolist())

def spectral_refined(cache):
    state,info=spectral(cache)
    return coordinate(cache,state)

def feedback(batches,initial_groups,scorer,start_maps,cap=10):
    groups=[z.copy() for z in initial_groups];maps=start_maps.copy();trace=[]
    score=float(sum(scorer.weights(b,maps)[z[b.i]==z[b.j]].sum() for b,z in zip(batches,groups)))
    trace.append(dict(iteration=-1,objective=score,map_changes=0,group_changes=0))
    converged=False
    for iteration in range(cap):
        changes=0
        for j,b in enumerate(batches):
            z=scorer.decode(b,maps);w=scorer.weights(b,maps)
            if w[z[b.i]==z[b.j]].sum()>w[groups[j][b.i]==groups[j][b.j]].sum()+1e-10:
                groups[j]=z;changes+=1
        cache=build_cache(pool(batches,groups),scorer,len(maps))
        state,info=exact(cache,cache.state(maps));new=cache.maps(state)
        nm=int(np.sum(new!=maps));maps=new
        if info['objective']<score-1e-7:raise AssertionError('Feedback decreased objective')
        score=info['objective'];trace.append(dict(iteration=iteration,objective=score,map_changes=nm,group_changes=changes))
        if not changes and not nm:converged=True;break
    return maps,dict(objective=score,cycles=iteration+1,converged=converged,trace=trace)

def self_checks():
    rng=np.random.default_rng(103_000_000);records=[]
    for K in [2,3,4,5]:
        S=3;perms=np.asarray(list(permutations(range(K))));hidden=np.vstack([np.arange(K),rng.permutation(K),rng.permutation(K)])
        scores={};counts={}
        for u,v in combinations(range(S),2):
            rel=np.argsort(hidden[v])[hidden[u]]
            scores[u,v]=(perms==rel).sum(axis=1).astype(float);counts[u,v]=1
        cache=cache_from_relative(K,S,scores,counts)
        for name,fn in [('exact',exact),('reference',reference_only),('spectral',spectral)]:
            state,_=fn(cache);assert np.array_equal(cache.maps(state),hidden),(K,name)
        noisy={edge:value+rng.uniform(-.1,.1,len(value)) for edge,value in scores.items()}
        ncache=cache_from_relative(K,S,noisy,counts)
        assert np.array_equal(ncache.maps(exact(ncache)[0]),hidden)
        records.append(dict(case='noiseless_conventions_and_bounded_perturbation',K=K,S=S,status='PASS',minimum_ideal_pair_margin=2,maximum_absolute_score_perturbation=.1))
    K=4;S=4;D=math.factorial(K);zeros={e:np.zeros(D) for e in combinations(range(S),2)}
    empty=cache_from_relative(K,S,zeros,{e:0 for e in zeros});a=empty.score_array()
    assert np.ptp(a)==0 and a.size==D**3
    records.append(dict(case='no_pair_information',optimal_states=a.size,status='PASS'))
    uniform=timing_batch(K,S)
    uniform=ObservableBatch(uniform.receiver,np.full_like(uniform.native,1/K),uniform.i,uniform.j,
        uniform.physical,uniform.time_z,uniform.freq_z,uniform.gate_k,uniform.kind)
    uniform_cache=build_cache(uniform,load_scorers(0,False)['joint'],S);a=uniform_cache.score_array()
    assert np.ptp(a)<1e-10 and int(np.sum(np.isclose(a,a.max(),rtol=0,atol=1e-10)))==D**3
    records.append(dict(case='actual_uniform_probability_vectors',optimal_states=a.size,status='PASS',objective_range=float(np.ptp(a))))
    perms=empty.permutations;hidden=np.vstack([np.arange(K),rng.permutation(K),rng.permutation(K),rng.permutation(K)])
    vals={};counts={}
    for u,v in combinations(range(S),2):
        vals[u,v]=np.zeros(D) if v==3 else (perms==np.argsort(hidden[v])[hidden[u]]).sum(axis=1).astype(float)
        counts[u,v]=0 if v==3 else 1
    disconnected=cache_from_relative(K,S,vals,counts);a=disconnected.score_array()
    assert int(np.sum(a==a.max()))==24
    records.append(dict(case='receiver_3_disconnected_from_reference_component',optimal_states=24,status='PASS'))
    hidden=np.vstack([rng.permutation(K) for _ in range(S)])
    vals={};counts={};path={(0,1),(1,2),(2,3)}
    for u,v in combinations(range(S),2):
        counts[u,v]=int((u,v) in path)
        vals[u,v]=(perms==np.argsort(hidden[v])[hidden[u]]).sum(axis=1).astype(float) if counts[u,v] else np.zeros(D)
    connected=cache_from_relative(K,S,vals,counts)
    for name,fn in [('exact',exact),('spectral',spectral)]:
        estimated=connected.maps(fn(connected)[0])
        # Optimization fixes native reference order. Restore its KNOWN names.
        restored=hidden[0][estimated]
        assert np.array_equal(restored,hidden),(name,restored,hidden)
    records.append(dict(case='connected_path_missing_direct_pairs_nonidentity_known_reference',status='PASS',known_reference=hidden[0].tolist()))
    return records

def run(replications=10,test_count=80):
    DEST.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    protocol=dict(replications=replications,test_scenes_per_replication=test_count,adaptation_budgets=[1,5,40],
        adaptation_seed='100000000+100000*r+j',test_seed='101000000+100000*r+j',hidden_map_seed='102000000+r',
        known_receiver=0,unknown_receivers=[1,2,3],anchors_at_unknown_receivers=0,classes=4,
        new_scenes_disjoint_from_all_retained_splits=True,original_linear_fits_frozen=True,
        boosted='Original training scenes and hyperparameters reconstructed; original stored temperature frozen',
        scoring_variants=['joint','unweighted','boosted'],methods=['unresolved','reference','spectral','coordinate','spectral_refined','exact','feedback','known'],
        physical_groups='Same frozen physical scorer for all map solvers and semantic scorers',
        coordinate_maximum_sweeps=100,feedback_maximum_sweeps=10,
        spectral_source='https://papers.nips.cc/paper/2013/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf',
        spectral_adaptation='Winner relative permutation per observed receiver pair; symmetric block matrix with identity diagonal; top-K eigenspace; reference-block Hungarian rounding',
        spectral_limitation='Pair-score margins are discarded; not the original RF score objective or official implementation',
        timings='cache_ms shared fixed-group preprocessing; solve_ms conditional map solve only; feedback includes additional group decoding and table rebuilding',
        selection='Fixed grids and configurations; no new test label selects parameters or methods',
        independent_recovery_claim='The joint map block is exact; the map/event problem remains approximate')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    checks=self_checks();rows=[];maps_rows=[];traces=[];manifests=[];legacy=[]
    for rep in range(replications):
        scorers=load_scorers(rep);rng=np.random.default_rng(102_000_000+rep)
        hidden=np.vstack([np.arange(4),*[rng.permutation(4) for _ in range(3)]])
        adapt_scenes=[simulate(100_000_000+100_000*rep+j,Config()) for j in range(40)]
        test_scenes=[simulate(101_000_000+100_000*rep+j,Config()) for j in range(test_count)]
        adapt=[controlled_adapter(s,hidden)[0] for s in adapt_scenes]
        test=[controlled_adapter(s,hidden) for s in test_scenes]
        identity=np.tile(np.arange(4),(4,1));groups=[scorers['physics'].decode(b,identity) for b in adapt]
        for split,scenes in [('adaptation',adapt_scenes),('test',test_scenes)]:
            manifests.extend(dict(replicate=rep,split=split,scene=j,seed=s.seed,reports=len(s.truth)) for j,s in enumerate(scenes))
        predictions={};boundaries=np.cumsum([0]+[len(s.truth) for s in test_scenes])
        for kind in protocol['scoring_variants']:
            scorer=scorers[kind];evaluation_cache={}
            def evaluate(maps):
                key=tuple(maps.ravel())
                if key not in evaluation_cache:
                    metrics=[];labels=[]
                    for scene,(b,p) in zip(test_scenes,test):
                        z=scorer.decode(b,maps);m=metric(scene,p,z)
                        assert m['receiver_conflicts']==m['gate_violations']==0
                        metrics.append(m);labels.append(z)
                    pred_key=f'{kind}_maps_{len(evaluation_cache)}'
                    predictions[pred_key]=np.concatenate(labels)
                    evaluation_cache[key]=(metrics,pred_key)
                return evaluation_cache[key]
            for budget in protocol['adaptation_budgets']:
                selected=pool(adapt[:budget],groups[:budget]);start=time.perf_counter()
                cache=build_cache(selected,scorer,4);cache_ms=(time.perf_counter()-start)*1000
                methods={}
                for name,fn in [('reference',reference_only),('spectral',spectral),('coordinate',coordinate),('spectral_refined',spectral_refined),('exact',exact)]:
                    start=time.perf_counter();state,info=fn(cache);elapsed=(time.perf_counter()-start)*1000
                    learned=cache.maps(state);direct=float(scorer.weights(selected,learned).sum())
                    assert abs(direct-info['objective'])<1e-7
                    methods[name]=(learned,{**info,'solve_ms':elapsed})
                start=time.perf_counter();learned,info=feedback(adapt[:budget],groups[:budget],scorer,methods['exact'][0]);elapsed=(time.perf_counter()-start)*1000
                for row in info.pop('trace'):traces.append(dict(replicate=rep,scorer=kind,budget=budget,**row))
                methods['feedback']=(learned,{**info,'solve_ms':elapsed})
                methods['known']=(hidden,dict(objective=cache.objective(cache.state(hidden)),solve_ms=0.,cycles=0,converged=True))
                methods['unresolved']=(identity,dict(objective=cache.objective(np.zeros(4,int)),solve_ms=0.,cycles=0,converged=True))
                for method,(learned,info) in methods.items():
                    m,pred_key=evaluate(learned);correct=int(np.sum(learned[1:]==hidden[1:]));fixed_objective=cache.objective(cache.state(learned))
                    rec=dict(replicate=rep,scorer=kind,budget=budget,method=method,correct=correct,hidden=12,
                        mapping_accuracy=correct/12,exact_map_block=bool(np.array_equal(learned,hidden)),
                        map=learned.tolist(),hidden_evaluation_only=hidden.tolist(),prediction_key=pred_key,
                        cache_ms=cache_ms,co_grouped_pairs=len(selected.i),fixed_group_objective=fixed_objective,
                        objective_gap=max(0.,methods['exact'][1]['objective']-fixed_objective),**info)
                    maps_rows.append(rec)
                    rows.extend(dict(replicate=rep,scorer=kind,budget=budget,method=method,scene=j,mapping_accuracy=correct/12,**x) for j,x in enumerate(m))
        np.savez_compressed(DEST/f'predictions_r{rep:02d}.npz',observation_boundaries=boundaries,**predictions)
        print(f'Map comparison {rep+1}/{replications}; {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'per_scene.csv.gz',index=False)
    r=frame.groupby(['replicate','scorer','budget','method'],as_index=False)[METRICS+['mapping_accuracy']].mean()
    r.to_csv(DEST/'per_replicate.csv',index=False)
    aggregate=summary(r,['scorer','budget','method'],METRICS)
    records=pd.DataFrame(maps_rows)
    m=records.groupby(['scorer','budget','method'],as_index=False).agg(correct=('correct','sum'),hidden=('hidden','sum'),mapping_accuracy=('mapping_accuracy','mean'),
        full_map_recoveries=('exact_map_block','sum'),median_cache_ms=('cache_ms','median'),median_solve_ms=('solve_ms','median'),mean_objective_gap=('objective_gap','mean'))
    aggregate.merge(m,on=['scorer','budget','method']).to_csv(DEST/'summary.csv',index=False)
    pd.DataFrame(manifests).to_csv(DEST/'scene_manifest.csv',index=False)
    pd.DataFrame(traces).to_csv(DEST/'feedback_traces.csv',index=False)
    (DEST/'learned_maps.json').write_text(json.dumps(maps_rows,indent=2),encoding='utf-8')
    (DEST/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    report=dict(status='PASS',seconds=time.perf_counter()-started,per_scene_rows=len(frame),map_fits=len(records),
        feedback_and_sequential_equal_maps=int(sum(np.array_equal(g[g.method=='feedback'].iloc[0]['map'],g[g.method=='exact'].iloc[0]['map']) for _,g in records.groupby(['replicate','scorer','budget']))),
        feedback_comparisons=replications*3*3,zero_receiver_and_gate_violations=bool((frame[['receiver_violation','gate_violation']]==0).all().all()),
        self_checks=len(checks),python=sys.version,platform=platform.platform(),numerical_threads=1)
    (DEST/'execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

def timing_batch(K,S,seed=104_000_000):
    rng=np.random.default_rng(seed+K*100+S);events=12
    receiver=np.repeat(np.arange(S),events);native=rng.dirichlet(np.ones(K),S*events)
    ii=[];jj=[]
    for u,v in combinations(range(S),2):
        ii.extend(u*events+np.arange(events));jj.extend(v*events+np.arange(events))
    physical=rng.uniform(0,2,(len(ii),5))
    return ObservableBatch(receiver,native,np.asarray(ii),np.asarray(jj),physical,np.zeros(len(ii)),np.zeros(len(ii)),3.5,'controlled')

def computational_checks():
    DEST.mkdir(parents=True,exist_ok=True);scorer=load_scorers(0,False)['joint'];rows=[]
    for K,U in [(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(6,1),(6,2)]:
        states=math.factorial(K)**U
        if states>MAX_STATES:
            try:build_cache(timing_batch(K,U+1),scorer,U+1)
            except ValueError:pass
            else:raise AssertionError('State guard failed')
            rows.append(dict(K=K,unknown_receivers=U,states=states,status='guard_rejected',score_array_MiB=states*8/2**20));continue
        b=timing_batch(K,U+1);build=[];solve=[]
        for repeat in range(3):
            start=time.perf_counter();cache=build_cache(b,scorer,U+1);build.append((time.perf_counter()-start)*1000)
            start=time.perf_counter();state,info=exact(cache);solve.append((time.perf_counter()-start)*1000)
            assert abs(scorer.weights(b,cache.maps(state)).sum()-info['objective'])<1e-7
        tracemalloc.start();cache=build_cache(b,scorer,U+1);state,_=exact(cache);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows.append(dict(K=K,unknown_receivers=U,states=states,status='evaluated',score_array_MiB=states*8/2**20,
            median_cache_ms=float(np.median(build)),median_solve_ms=float(np.median(solve)),python_traced_peak_MiB=peak/2**20,
            pair_terms=len(cache.tables),co_grouped_pairs=len(b.i),fixed_pairs_per_receiver_pair=12))
    pd.DataFrame(rows).to_csv(DEST/'computational_scaling.csv',index=False)
    hidden=np.vstack([np.arange(4),np.random.default_rng(102_000_000).permutation(4),np.array([1,3,0,2]),np.array([3,2,1,0])])
    scenes=[simulate(100_000_000+j,Config()) for j in range(5)]
    batches=[controlled_adapter(s,hidden)[0] for s in scenes];identity=np.tile(np.arange(4),(4,1))
    physical=load_scorers(0,False)['physics'];groups=[physical.decode(b,identity) for b in batches]
    batch=pool(batches,groups);rows=[]
    for repeat in range(3):
        start=time.perf_counter();cache=build_cache(batch,scorer,4);state,info=exact(cache);cached=(time.perf_counter()-start)*1000
        start=time.perf_counter();best=-np.inf;best_maps=None
        for indices in product(range(24),repeat=3):
            maps=cache.maps([0,*indices]);value=float(scorer.weights(batch,maps).sum())
            if value>best:best=value;best_maps=maps
        uncached=(time.perf_counter()-start)*1000
        assert abs(best-info['objective'])<1e-7
        rows.append(dict(repeat=repeat,states=cache.states,adaptation_scenes=5,co_grouped_pairs=len(batch.i),
            cached_total_ms=cached,uncached_total_ms=uncached,speedup=uncached/cached,
            objective_difference=abs(best-info['objective']),max_distinct_pair_scores=6*24,
            uncached_full_objective_calls=cache.states))
    pd.DataFrame(rows).to_csv(DEST/'cached_vs_uncached.csv',index=False)
    (DEST/'computational_protocol.json').write_text(json.dumps(dict(
        purpose='Computational cost only; K-dependent probability vectors are not a new RF accuracy dataset',
        scaler_and_coefficients='Frozen weighted linear r00; K enters compatibility and entropy normalization',
        groups='12 arbitrary co-grouped report pairs for every receiver pair',
        repeats=3,times='Medians; tracemalloc disabled during timing',
        memory='Separate measured Python-traced allocation peak; not process RSS and excludes untraced native library allocations',
        score_array_memory='Arithmetic 8*Q bytes, not a measurement',
        state_limit=MAX_STATES,cache_comparison='Same five new adaptation scenes; full weighted RF objective reevaluated in every uncached joint state',
        no_accuracy_or_large_vocabulary_recovery_claim=True),indent=2),encoding='utf-8')
    print('Computational checks complete',flush=True)

def hard_cases():
    from deep_semantic_stress import make_scene
    destination=DEST/'semantic_stress';destination.mkdir(parents=True,exist_ok=True)
    conditions=['same_class_collisions','dense_same_class_collisions']
    protocol=dict(conditions=conditions,replications=10,adaptation_scenes=5,test_scenes_per_replication=40,
        adaptation_seed='112000000+100000*r+j',test_seed='113000000+100000*r+j',hidden_seed='114000000+r',
        known_receiver=0,unknown_receivers=[1,2,3],anchors=0,classes=4,
        scorers=['joint','unweighted','boosted'],methods=['unresolved','reference','spectral','coordinate','exact','feedback','known'],
        scorer_parameters_and_interventions_frozen=True,test_labels_used_for_selection=False,
        purpose='Connect same-class collision stress to unknown map estimation and event association',
        scenario_source='code/deep_semantic_stress.py; original and dense collision probabilities0.25 and0.65',
        relationship_to_nominal='Independent newly generated seeds; no comparisons use nominal historical test labels')
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];records=[];traces=[];started=time.perf_counter()
    for rep in range(10):
        scorers=load_scorers(rep);rng=np.random.default_rng(114_000_000+rep)
        hidden=np.vstack([np.arange(4),*[rng.permutation(4) for _ in range(3)]])
        for condition in conditions:
            adapt_scenes=[make_scene(112_000_000+100_000*rep+j,condition)[0] for j in range(5)]
            test_scenes=[make_scene(113_000_000+100_000*rep+j,condition)[0] for j in range(40)]
            batches=[controlled_adapter(s,hidden)[0] for s in adapt_scenes]
            tests=[controlled_adapter(s,hidden) for s in test_scenes]
            identity=np.tile(np.arange(4),(4,1));groups=[scorers['physics'].decode(b,identity) for b in batches]
            selected=pool(batches,groups);predictions={};boundaries=np.cumsum([0]+[len(s.truth) for s in test_scenes])
            for kind in protocol['scorers']:
                scorer=scorers[kind];cache=build_cache(selected,scorer,4);methods={};evaluation_cache={}
                for name,fn in [('reference',reference_only),('spectral',spectral),('coordinate',coordinate),('exact',exact)]:
                    state,info=fn(cache);methods[name]=(cache.maps(state),info)
                learned,info=feedback(batches,groups,scorer,methods['exact'][0])
                for step in info.pop('trace'):traces.append(dict(replicate=rep,condition=condition,scorer=kind,**step))
                methods['feedback']=(learned,info)
                methods['known']=(hidden,dict(objective=cache.objective(cache.state(hidden))))
                methods['unresolved']=(identity,dict(objective=cache.objective(np.zeros(4,int))))
                for method,(maps,info) in methods.items():
                    key=tuple(maps.ravel())
                    if key not in evaluation_cache:
                        m=[];labels=[]
                        for s,(b,p) in zip(test_scenes,tests):
                            z=scorer.decode(b,maps);entry=metric(s,p,z);m.append(entry);labels.append(z)
                            assert entry['receiver_conflicts']==entry['gate_violations']==0
                        pred_key=f'{kind}_{len(evaluation_cache)}';predictions[pred_key]=np.concatenate(labels)
                        evaluation_cache[key]=(m,pred_key)
                    m,pred_key=evaluation_cache[key];correct=int(np.sum(maps[1:]==hidden[1:]))
                    records.append(dict(replicate=rep,condition=condition,scorer=kind,method=method,correct=correct,hidden=12,
                        mapping_accuracy=correct/12,map=maps.tolist(),hidden_evaluation_only=hidden.tolist(),prediction_key=pred_key,
                        fixed_group_objective=cache.objective(cache.state(maps)),**info))
                    rows.extend(dict(replicate=rep,condition=condition,scorer=kind,method=method,scene=j,mapping_accuracy=correct/12,**x) for j,x in enumerate(m))
            np.savez_compressed(destination/f'predictions_{condition}_r{rep:02d}.npz',observation_boundaries=boundaries,**predictions)
        print(f'Map semantic stress {rep+1}/10; {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(destination/'per_scene.csv.gz',index=False)
    rep=frame.groupby(['replicate','condition','scorer','method'],as_index=False)[METRICS+['mapping_accuracy']].mean();rep.to_csv(destination/'per_replicate.csv',index=False)
    aggregate=summary(rep,['condition','scorer','method'],METRICS)
    rec=pd.DataFrame(records);m=rec.groupby(['condition','scorer','method'],as_index=False).agg(correct=('correct','sum'),hidden=('hidden','sum'),mapping_accuracy=('mapping_accuracy','mean'))
    aggregate.merge(m,on=['condition','scorer','method']).to_csv(destination/'summary.csv',index=False)
    pd.DataFrame(traces).to_csv(destination/'feedback_traces.csv',index=False)
    (destination/'learned_maps.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    report=dict(status='PASS',seconds=time.perf_counter()-started,scene_rows=len(frame),map_fits=len(rec),
        feedback_vs_sequential_same_maps=int(sum(np.array_equal(g[g.method=='feedback'].iloc[0]['map'],g[g.method=='exact'].iloc[0]['map']) for _,g in rec.groupby(['replicate','condition','scorer']))),feedback_comparisons=60)
    (destination/'execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

def artifacts():
    from scipy.stats import t as student_t
    d=pd.read_csv(DEST/'summary.csv');r=pd.read_csv(DEST/'per_replicate.csv')
    hard=pd.read_csv(DEST/'semantic_stress/summary.csv')
    names={'joint':'Weighted linear','unweighted':'Unweighted linear','boosted':'Boosted'}
    method_names={'unresolved':'Unresolved','reference':'Reference only','spectral':'Spectral sync.','coordinate':'Coordinate','spectral_refined':'Spectral + coordinate','exact':'Exact block','feedback':'Alternating','known':'Known maps'}
    table=['\\begin{tabularx}{\\linewidth}{@{}lYrr@{}}\\toprule',
           'Scorer & Map estimator & Correct / hidden & Event-pair F1 [95\\% CI] \\\\\\midrule']
    for kind in ['joint','unweighted','boosted']:
        for method in ['unresolved','reference','spectral','spectral_refined','coordinate','exact']:
            x=d[(d.scorer==kind)&(d.budget==5)&(d.method==method)].iloc[0]
            table.append(f"{names[kind]} & {method_names[method]} & {int(x.correct)}/{int(x.hidden)} & {x.f1:.4f} [{x.f1_lo:.4f}, {x.f1_hi:.4f}] \\\\")
        table.append('\\addlinespace')
    table.append('\\bottomrule\\end{tabularx}')
    (ROOT/'tables/deep_mapping_methods.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    table=['\\begin{tabularx}{\\linewidth}{@{}Yrrrrr@{}}\\toprule',
           'Scorer & Adapt. & Exact map & Spectral map & Reference map & Exact F1 \\\\\\midrule']
    for kind in ['joint','unweighted','boosted']:
        for budget in [1,5,40]:
            q=d[(d.scorer==kind)&(d.budget==budget)].set_index('method')
            table.append(f"{names[kind]} & {budget} & {int(q.loc['exact','correct'])}/120 & {int(q.loc['spectral','correct'])}/120 & {int(q.loc['reference','correct'])}/120 & {q.loc['exact','f1']:.4f} \\\\")
    table.append('\\bottomrule\\end{tabularx}')
    (ROOT/'tables/deep_mapping_budget.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    table=['\\begin{tabularx}{\\linewidth}{@{}lYrrrrr@{}}\\toprule',
           'Copy prob. & Scorer & Map: spectral / exact & Known F1 & Unresolved F1 & Spectral F1 & Exact F1 \\\\\\midrule']
    for condition,prob in [('same_class_collisions',.25),('dense_same_class_collisions',.65)]:
        for kind in ['joint','unweighted','boosted']:
            q=hard[(hard.scorer==kind)&(hard.condition==condition)].set_index('method')
            table.append(f"{prob:.2f} & {names[kind]} & {int(q.loc['spectral','correct'])}/120 / {int(q.loc['exact','correct'])}/120 & {q.loc['known','f1']:.4f} & {q.loc['unresolved','f1']:.4f} & {q.loc['spectral','f1']:.4f} & {q.loc['exact','f1']:.4f} \\\\")
    table.append('\\bottomrule\\end{tabularx}')
    (ROOT/'tables/deep_mapping_stress.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    scaling=pd.read_csv(DEST/'computational_scaling.csv')
    table=['\\begin{tabularx}{\\linewidth}{@{}rrrYrr@{}}\\toprule',
           '$K$ & $U$ & Joint states & Pair-table ms & Search ms & Traced peak MiB \\\\\\midrule']
    for _,x in scaling.iterrows():
        if (int(x.K),int(x.unknown_receivers)) not in [(2,3),(3,3),(4,3),(4,4),(5,1),(5,2),(5,3),(6,1),(6,2)]:continue
        if x.status=='guard_rejected':
            table.append(f"{int(x.K)} & {int(x.unknown_receivers)} & {int(x.states):,} & Guard rejected & -- & -- \\\\")
        else:
            table.append(f"{int(x.K)} & {int(x.unknown_receivers)} & {int(x.states):,} & {x.median_cache_ms:.3f} & {x.median_solve_ms:.3f} & {x.python_traced_peak_MiB:.3f} \\\\")
    table.append('\\bottomrule\\end{tabularx}')
    (ROOT/'tables/deep_mapping_cost.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    differences=[]
    for (kind,budget),g in r.groupby(['scorer','budget']):
        pivot=g.pivot(index='replicate',columns='method',values='f1')
        for comparator in ['spectral','coordinate','reference','spectral_refined','feedback']:
            delta=pivot['exact']-pivot[comparator];half=student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            differences.append(dict(scorer=kind,budget=int(budget),comparison='exact_minus_'+comparator,replications=len(delta),
                mean=float(delta.mean()),lo=float(delta.mean()-half),hi=float(delta.mean()+half)))
    pd.DataFrame(differences).to_csv(DEST/'paired_differences.csv',index=False)
    cost=pd.read_csv(DEST/'cached_vs_uncached.csv')
    notes='''Executed map-completion study (targeted revision)\n\nAll tests preserve the original RF scope. We infer missing permutations of a declared signal-family vocabulary. Receiver 0 supplies its known class names. Three receivers have entirely unknown four-class maps. The scoring and physical grouping fits remain frozen.\n\nThe main comparison uses 400 new adaptation scenes and 800 new test scenes across ten replications. Adaptation budgets are 1, 5 and 40. The budgets use prefixes of each independent adaptation set. Every budget is reported. New test references select no score, map update or stopping point. The weighted and unweighted linear fits are read unchanged. The boosted model is reconstructed from its original 120 training scenes and original hyperparameters. Its saved original temperature is reused. The replication-0 probability vector on all original 80 test scenes agrees exactly (maximum difference 0).\n\nAll solvers start from the same physical-only provisional groups. Reference-only fitting uses only edges to receiver 0. Coordinate updates begin with the same identity completion and use up to 100 strict-improvement sweeps. The adapted spectral baseline first chooses the best relative permutation for each observed receiver pair. It places these permutations in a symmetric block matrix with identity diagonal. Its top-K eigenspace is projected to reference blocks. Hungarian rounding recovers each map. This is Algorithm 1 of Pachauri, Kondor and Singh, NIPS 2013, with RF-specific pair-map estimation. It discards score margins. It is not their original implementation or an optimization of the full nonlinear RF score objective. Spectral-plus-coordinate uses the same RF objective for a subsequent polish. Exact enumeration keeps all candidate scores and solves the coupled fixed-group block.\n\nAt five adaptation scenes, exact enumeration recovers 120/120 entries for each scorer. New-test F1 is 0.7786 weighted, 0.7997 unweighted and 0.8232 boosted. The corresponding spectral results are 114/120 and 0.7712; 120/120 and 0.7997; and 114/120 and 0.8134. At forty scenes, reference-only, spectral and exact all recover every hidden entry. The results do not establish a universal gain over permutation synchronization. The exact-vs-spectral paired intervals are recorded separately. Coordinate updates retain poorer configurations even with more data. Sequential exact and alternating exact return the same maps in all 90 configurations.\n\nTwo additional same-class collision experiments use five adaptation scenes, 40 test scenes per replication and disjoint seeds. Exact mapping recovers all 120 entries for each scorer at collision-copy probability 0.25 and 0.65. Yet event F1 at 0.65 remains 0.4450, 0.4504 and 0.4537. These match each scorer's known-map reference. Correct class registration cannot identify different transmissions of the same class. Spectral and exact tie throughout the dense condition. Feedback maps equal sequential exact maps in all 60 additional configurations.\n\nThe eight executable checks cover K=2..5 noiseless recovery, bounded score perturbations, all-flat score ambiguity, a disconnected source, and a connected path with missing direct-reference edges and a nonidentity known reference map. In the all-flat test all 13,824 states maximize the objective. A disconnected unknown receiver permits 24 equally optimal states. These are mathematical score-table checks, not RF accuracy datasets.\n\nComputational scaling varies K and the number of unknown receivers with 12 arbitrary fixed-group pairs per receiver pair. K!=4 uses probability vectors and the existing six-feature calculation only to time the computational kernel. It supplies no larger-vocabulary RF recovery evidence. Timings are medians of three runs. Memory is a separate measured Python-traced allocation peak, not process RSS. Arithmetic dense-score storage is also saved. The one-million-state guard rejects K=5,U=3.\n\nAll new event partitions meet the receiver and gate constraints. Mapping accuracy counts 120 entries from ten map interventions; entries are not independent Bernoulli trials. Event-F1 confidence intervals use ten replication means. Original figures, tables, results, methods and manuscript sources were not edited by this subtask.\n'''
    notes+=f"\nCache comparison: median {cost.cached_total_ms.median():.4f} ms versus {cost.uncached_total_ms.median():.4f} ms for reevaluation of all 13,824 complete map states. This uses one fixed five-scene adaptation sample, the weighted scorer, 160 co-grouped pairs and three timing repeats. Every objective optimum agrees exactly. This is an implementation-specific speed comparison.\n"
    (ROOT/'deep_revision_audit/mapping_notes.txt').write_text(notes,encoding='utf-8')
    print('Mapping tables and notes written')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all','mapping','cost','check','stress','artifacts'],default='all')
    parser.add_argument('--replications',type=int,default=10);parser.add_argument('--test-scenes',type=int,default=80);args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.task=='check':
            checks=self_checks();DEST.mkdir(parents=True,exist_ok=True)
            (DEST/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
            if (DEST/'execution.json').exists():
                report=json.loads((DEST/'execution.json').read_text());report['self_checks']=len(checks)
                (DEST/'execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(checks,indent=2))
        if args.task in ['all','mapping']:run(args.replications,args.test_scenes)
        if args.task in ['all','cost']:computational_checks()
        if args.task in ['all','stress']:hard_cases()
        if args.task in ['all','artifacts']:artifacts()
