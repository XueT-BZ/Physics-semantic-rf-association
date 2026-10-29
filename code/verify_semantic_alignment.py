"""Recompute new alignment metrics independently from saved labels and truth."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from experiment import Config,simulate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/semantic_alignment'


def choose_two(counts):
    return sum(v*(v-1)//2 for v in counts.values())


def main():
    frame=pd.read_csv(OUT/'per_scene.csv').set_index(['replicate','scene','condition'])
    checked=0;reencoded=0
    for rep in range(10):
        z=np.load(OUT/f'predictions_r{rep:02d}.npz')
        old=np.load(ROOT/f'results/factorial/predictions_r{rep:02d}.npz')
        assert np.array_equal(z['mapped'],old['joint_constrained'])
        assert np.array_equal(z['mapped'],z['representation_reencoded'])
        for scene_id in range(80):
            scene=simulate(1_000_000+100_000*rep+20_000+scene_id,Config())
            a,b=z['observation_boundaries'][scene_id:scene_id+2]
            assert b-a==len(scene.truth)
            reencoded+=1
            for condition in ['mapped','native_order_bypassed','class_evidence_neutral']:
                pred=z[condition][a:b];truth=scene.truth
                tp=choose_two(Counter(zip(truth.tolist(),pred.tolist())))
                total_pred=choose_two(Counter(pred.tolist()))
                total_true=choose_two(Counter(truth.tolist()))
                fp=total_pred-tp;fn=total_true-tp
                values={'precision':tp/total_pred if total_pred else 0.,
                        'recall':tp/total_true if total_true else 0.,
                        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.}
                row=frame.loc[(rep,scene_id,condition)]
                for metric,value in values.items():assert np.isclose(row[metric],value,atol=1e-12)
                for label in np.unique(pred):
                    members=np.flatnonzero(pred==label)
                    assert len(np.unique(scene.obs['receiver'][members]))==len(members)
                checked+=1
    means=frame.reset_index().groupby(['replicate','condition']).f1.mean()
    summary=pd.read_csv(OUT/'summary.csv').set_index('condition')
    for condition in summary.index:
        values=means.xs(condition,level='condition').to_numpy()
        mean=float(values.mean());error=float(t.ppf(.975,9)*values.std(ddof=1)/np.sqrt(10))
        for name,value in [('f1',mean),('f1_lo',mean-error),('f1_hi',mean+error)]:
            assert np.isclose(summary.loc[condition,name],value,atol=1e-12)
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['script_sha256']==hashlib.sha256((ROOT/'code/semantic_alignment_audit.py').read_bytes()).hexdigest()
    report=dict(status='PASS',scene_conditions_independently_recomputed=checked,
        matched_reencoding_partitions_verified=reencoded,
        full_mapping_matches_released_joint_predictions=True,replication_intervals_recomputed=True,
        scope='Independent saved-label metric audit; known mappings only.')
    assert checked==2400 and reencoded==800
    target=ROOT/'audit/verification/semantic_alignment.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
