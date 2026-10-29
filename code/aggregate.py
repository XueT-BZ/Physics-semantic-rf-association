#!/usr/bin/env python3
"""Combine independently executed complete replications; never invent missing runs."""
from pathlib import Path
import argparse, json, shutil
import numpy as np
import pandas as pd
from scipy.stats import t
from experiment import summarize

def aggregate(root: Path, out: Path, reps: int = 10) -> None:
    dirs = [root / f'{i:02d}' for i in range(reps)]
    for d in dirs:
        if not (d / 'environment.json').exists():
            raise FileNotFoundError(f'Incomplete replication: {d}')
    out.mkdir(parents=True, exist_ok=True)
    names = ['pair_per_replicate', 'cluster_per_scene', 'cluster_per_replicate',
             'stress_per_replicate', 'gate_per_replicate', 'exact_cases',
             'temperatures', 'split_manifest']
    frames = {}
    for name in names:
        df = pd.concat([pd.read_csv(d/f'{name}.csv') for d in dirs], ignore_index=True)
        if 'replicate' in df and set(df.replicate) != set(range(reps)):
            raise ValueError(f'Missing or duplicated run identifiers in {name}')
        df.to_csv(out/f'{name}.csv', index=False)
        frames[name] = df
    for d in dirs:
        for pat in ['observations_*.csv.gz', 'linear_models_*.json', 'prediction_*.npz', 'example_scene.npz']:
            for f in d.glob(pat):
                shutil.copy2(f, out/f.name)
    pm = ['auroc','ap','brier','nll','ece','admitted_ece','positive_rate']
    cm = ['f1','precision','recall','ari','b3f1','exact','receiver_violation','gate_violation','clutter_attachment','runtime_ms']
    sm = ['f1','precision','recall','ari','exact','runtime_ms','observations','gate_recall','retained_fraction','auroc','ap','brier','nll']
    summarize(frames['pair_per_replicate'], ['method'], pm).to_csv(out/'pair_summary.csv', index=False)
    summarize(frames['cluster_per_replicate'], ['method'], cm).to_csv(out/'cluster_summary.csv', index=False)
    summarize(frames['stress_per_replicate'], ['group','value','method'], sm).to_csv(out/'stress_summary.csv', index=False)
    summarize(frames['gate_per_replicate'], ['k'], ['f1','gate_recall','retained_fraction']).to_csv(out/'gate_summary.csv', index=False)
    delta=[]
    comparisons = [('pair','physics_calibrated','joint_calibrated','ap'),
        ('pair','joint_gated','joint_calibrated','brier'),
        ('cluster','physics_constrained','joint_constrained','f1'),
        ('cluster','joint_components','joint_constrained','f1'),
        ('cluster','boosted_constrained','joint_constrained','f1'),
        ('cluster','unweighted_semantics','joint_constrained','f1')]
    for group,base,target,metric in comparisons:
        df=frames['pair_per_replicate' if group=='pair' else 'cluster_per_replicate']
        wide=df.pivot(index='replicate',columns='method',values=metric)
        v=wide[target]-wide[base]
        h=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
        delta.append(dict(group=group,metric=metric,baseline=base,target=target,
                          delta=v.mean(),lo=v.mean()-h,hi=v.mean()+h))
    pd.DataFrame(delta).to_csv(out/'paired_differences.csv',index=False)
    envs=[json.loads((d/'environment.json').read_text()) for d in dirs]
    assert len({e['script_sha256'] for e in envs})==1, 'Different code versions'
    env=envs[0].copy()
    env.update(replications=reps,elapsed_seconds=sum(e['elapsed_seconds'] for e in envs),
               execution='Independent sequential single-replication runs, then aggregation',
               replicate_elapsed_seconds=[e['elapsed_seconds'] for e in envs])
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print('Aggregated',reps,'complete replications into',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reps',type=int,default=10)
    a=p.parse_args();aggregate(a.runs,a.out,a.reps)
