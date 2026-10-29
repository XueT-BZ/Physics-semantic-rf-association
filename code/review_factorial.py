"""Recompute joint accuracy/feasibility tables from archived frozen predictions."""
import json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import features,cluster_metrics
from review_common import ROOT,OUT,test_scenes,metric,summary,environment,METRICS

DEST=OUT/'factorial'

def main():
    DEST.mkdir(parents=True,exist_ok=True);environment('factorial');started=time.perf_counter()
    old=pd.read_csv(ROOT/'results/factorial/per_scene.csv').set_index(['replicate','scene','scorer','decoder'])
    rows=[];maximum=0.;checks=0
    for rep in range(10):
        archive=np.load(ROOT/f'results/factorial/predictions_r{rep:02d}.npz')
        boundaries=archive['observation_boundaries']
        for index,scene in enumerate(test_scenes(rep)):
            p=features(scene);lo,hi=boundaries[index:index+2]
            assert hi-lo==len(scene.truth)
            for scorer in ['physics','joint','unweighted','boosted']:
                for decoder in ['components','assignment','constrained']:
                    z=archive[scorer+'_'+decoder][lo:hi];m=metric(scene,p,z)
                    reference=old.loc[(rep,index,scorer,decoder)]
                    for key in METRICS:
                        difference=abs(m[key]-float(reference[key]));maximum=max(maximum,difference)
                        assert difference<1e-12,(rep,index,scorer,decoder,key,difference)
                    if index<2:
                        independent=cluster_metrics(scene,p,z)
                        for key in METRICS:assert abs(m[key]-independent[key])<1e-12,key
                        checks+=1
                    rows.append(dict(replicate=rep,scene=index,scorer=scorer,decoder=decoder,**m))
        print(f'Full metric audit {rep+1}/10',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'per_scene.csv.gz',index=False)
    grouped=frame.groupby(['replicate','scorer','decoder'],as_index=False)[METRICS].mean()
    grouped.to_csv(DEST/'per_replicate.csv',index=False)
    summary(grouped,['scorer','decoder'],METRICS).to_csv(DEST/'summary.csv',index=False)
    # Pair counts are pooled separately; the manuscript reports scene-macro F1.
    totals=frame.groupby(['replicate','scorer','decoder'],as_index=False)[['tp','fp','fn','true_pairs','predicted_pairs']].sum()
    totals['pair_micro_f1']=2*totals.tp/(2*totals.tp+totals.fp+totals.fn)
    totals.to_csv(DEST/'pooled_pair_counts.csv',index=False)
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',rows=len(frame),independent_legacy_metric_checks=checks,
        maximum_difference_from_archived_metrics=maximum,seconds=time.perf_counter()-started),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
