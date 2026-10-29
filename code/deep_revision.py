"""Reproduce or independently verify the targeted semantic-alignment studies."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]

def launch(script,*arguments):
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,str(ROOT/'code'/script),*arguments],cwd=ROOT,env=env,check=True)

def verify():
    import numpy as np
    import pandas as pd
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    from experiment import Config,simulate,features
    from review_common import metric,slice_scene
    start=time.perf_counter();out=ROOT/'results/deep_revision'
    before=json.loads((ROOT/'deep_revision_audit/source_manifest.json').read_text())
    protected={k:v for k,v in before.items() if k.split('/')[0] in {'code','data','figures','results','tables'}}
    for relative,expected in protected.items():
        path=ROOT/relative
        assert path.is_file(),f'Missing retained evidence: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'Changed retained evidence: {relative}'
    checked=0;numeric_max=0.
    def check(saved,current):
        nonlocal checked,numeric_max
        for key in ['tp','fp','fn','f1','ari','b3f1','exact','receiver_conflicts','gate_violations']:
            delta=abs(float(saved[key])-float(current[key]));numeric_max=max(numeric_max,delta)
            assert delta<1e-10,(key,saved[key],current[key])
        assert current['receiver_conflicts']==current['gate_violations']==0
        checked+=1
    with threadpool_limits(limits=1):
        frame=pd.read_csv(out/'semantic_stress/per_scene.csv')
        assert len(frame)==12800
        with np.load(out/'semantic_stress/predictions.npz') as predictions:
            for (rep,index),part in frame.groupby(['replication','scene']):
                seed=110_000_000+100_000*int(rep)+int(index)
                ordinary=simulate(seed,Config());dense=simulate(seed,replace(Config(),collisions=.65))
                scenes={False:(ordinary,features(ordinary)),True:(dense,features(dense))}
                for row in part.itertuples(index=False):
                    scene,p=scenes[row.condition.startswith('dense_')]
                    z=predictions[f'r{rep}_{index}_{row.condition}_{row.scorer}']
                    check(row._asdict(),metric(scene,p,z))
        print(f'Verified {checked} semantic partitions.',flush=True)
        for folder,hard in [('mapping',False),('mapping/semantic_stress',True)]:
            base=out/folder;frame=pd.read_csv(base/'per_scene.csv.gz')
            maps=json.loads((base/'learned_maps.json').read_text())
            groupkeys=['replicate','condition'] if hard else ['replicate']
            md=pd.DataFrame(maps)
            keys=['replicate']+(['condition'] if hard else [])+['scorer']+([] if hard else ['budget'])+['method','scene']
            lookup=frame.set_index(keys)
            for key,records in md.groupby(groupkeys):
                if not isinstance(key,tuple):key=(key,)
                rep=int(key[0]);condition=str(key[1]) if hard else 'nominal'
                count=40 if hard else 80;seedbase=113_000_000 if hard else 101_000_000
                cfg=replace(Config(),collisions=.65) if condition.startswith('dense_') else Config()
                scenes=[simulate(seedbase+100_000*rep+j,cfg) for j in range(count)]
                pairs=[features(s) for s in scenes]
                filename=f'predictions_{condition}_r{rep:02d}.npz' if hard else f'predictions_r{rep:02d}.npz'
                cache={}
                with np.load(base/filename) as predictions:
                    bounds=predictions['observation_boundaries']
                    assert np.array_equal(bounds,np.r_[0,np.cumsum([len(s.truth) for s in scenes])])
                    for row in records.to_dict('records'):
                        estimated=np.asarray(row['map']);hidden=np.asarray(row['hidden_evaluation_only'])
                        assert np.array_equal(estimated[0],hidden[0])
                        assert all(np.array_equal(np.sort(x),np.arange(4)) for x in estimated)
                        assert int(np.sum(estimated[1:]==hidden[1:]))==row['correct']
                        predkey=row['prediction_key']
                        if predkey not in cache:
                            labels=predictions[predkey]
                            cache[predkey]=[metric(s,p,labels[bounds[j]:bounds[j+1]]) for j,(s,p) in enumerate(zip(scenes,pairs))]
                        for j,current in enumerate(cache[predkey]):
                            query=(rep,)+((condition,) if hard else ())+(row['scorer'],)+(() if hard else (row['budget'],))+(row['method'],j)
                            check(lookup.loc[query].to_dict(),current)
            print(f'Verified {folder}: {len(frame)} saved partitions.',flush=True)
        frame=pd.read_csv(out/'runtime/per_instance.csv')
        with np.load(out/'runtime/predictions.npz') as predictions:
            for (rep,n),part in frame.groupby(['replicate','observations']):
                row=part.iloc[0];source=simulate(int(row.scene_seed),replace(Config(),events=int(row.source_event_count)))
                s=slice_scene(source,predictions[f'{rep}_{n}_source_indices']);p=features(s)
                for row in part.itertuples(index=False):
                    if row.has_feasible_output:check(row._asdict(),metric(s,p,predictions[f'{rep}_{n}_{row.method}']))
                    if row.method=='milp':assert row.proven_optimal and row.solver_status==0
        original=pd.read_csv(ROOT/'results/loed/observations.csv.gz')
        ids=set(original[original.split=='test'].observation_id)
        for method in ['supervised_reference','training_metadata_mode']:
            observed=set()
            for day in ['2020-08-03','2020-09-02']:
                new=pd.read_csv(out/f'loed/prediction_{method}_{day}.csv.gz').sort_values('observation_id')
                old=pd.read_csv(ROOT/f'results/loed/prediction_{day}_metadata_constrained.csv.gz').sort_values('observation_id')
                assert new.observation_id.tolist()==old.observation_id.tolist()
                oldlabel='predicted_cluster'
                assert np.array_equal(pd.factorize(new.predicted,sort=False)[0],pd.factorize(old[oldlabel],sort=False)[0])
                observed.update(new.observation_id)
            assert observed==ids and len(ids)==23677
    result=dict(status='PASS',retained_scientific_files_unchanged=len(protected),
        recomputed_partition_rows=checked,maximum_metric_difference=numeric_max,
        public_test_reports=23677,public_partitions_match_retained_results=True,
        seconds=time.perf_counter()-start,
        scope='All retained new partition labels checked against regenerated descriptor truth; old scientific evidence hashes verified.')
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all','mapping','semantic','public','runtime','verify'],default='verify')
    task=parser.parse_args().task
    if task in ['all','mapping']:launch('deep_mapping.py','--task','all');launch('deep_scoring.py')
    if task in ['all','semantic']:launch('deep_semantic_stress.py');launch('deep_semantic_artifacts.py')
    if task in ['all','public']:launch('deep_loed_audit.py')
    if task in ['all','runtime']:launch('deep_runtime.py')
    if task in ['all','verify']:verify()

if __name__=='__main__':main()
