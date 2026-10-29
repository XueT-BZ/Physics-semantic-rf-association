"""Cross four frozen scorers with three decoders on the existing 800 test scenes.

Linear parameters are read from the released files. The unarchived boosted model
is refitted on the original training/calibration scenes, with replication-0
probabilities checked against the released predictions. No new scenes or labels
are added. All decoders within a scorer receive identical probabilities/gates.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from experiment import (Config, simulate, features, Scorer, constrained_cluster,
                        connected_cluster, cluster_metrics, summarize)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/factorial'
KINDS = ['physics', 'joint', 'unweighted', 'boosted']
DECODERS = ['components', 'assignment', 'constrained']
METRICS = ['f1','precision','recall','ari','exact','receiver_violation','gate_violation']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = pd.read_csv(ROOT/'results/cluster_per_scene.csv')
    agreement = []
    started = time.time()
    for rep in range(10):
        seed = 1_000_000 + rep*100_000
        params = json.loads((ROOT/f'results/linear_models_r{rep:02d}.json').read_text())
        train = [features(simulate(seed+k, Config())) for k in range(120)]
        cal = [features(simulate(seed+10_000+k, Config())) for k in range(40)]
        boosted = Scorer('boosted')
        boosted.fit(train)
        boosted.calibrate(cal)
        saved_labels = {f'{k}_{d}': [] for k in KINDS for d in DECODERS}
        saved_scores = {k: [] for k in KINDS}
        boundaries = [0]
        pair_boundaries = [0]
        for index in range(80):
            scene = simulate(seed+20_000+index, Config())
            pairs = features(scene)
            boundaries.append(boundaries[-1]+len(scene.truth))
            pair_boundaries.append(pair_boundaries[-1]+len(pairs.y))
            for kind in KINDS:
                if kind == 'boosted':
                    prob = boosted.score(pairs, True, False)
                else:
                    p = params[kind]
                    design = pairs.Xunweighted if kind == 'unweighted' else pairs.X
                    x = design[:, p['columns']]
                    logits = ((x-np.array(p['mean']))/np.array(p['scale']))@np.array(p['coefficient'])+p['intercept']
                    prob = expit(logits/p['temperature'])
                saved_scores[kind].append(prob)
                decoded = {
                    'components': connected_cluster(scene, pairs, prob),
                    'assignment': connected_cluster(scene, pairs, prob, assignment=True),
                    'constrained': constrained_cluster(scene, pairs, prob)}
                for decoder, z in decoded.items():
                    metric = cluster_metrics(scene, pairs, z)
                    if decoder == 'constrained':
                        assert metric['receiver_violation'] == 0 and metric['gate_violation'] == 0
                        original_kind = {'physics':'physics_constrained','joint':'joint_constrained',
                                         'unweighted':'unweighted_semantics','boosted':'boosted_constrained'}[kind]
                        old = reference[(reference.replicate == rep)&(reference.scene == index)&(reference.method == original_kind)].iloc[0]
                        assert np.isclose(metric['f1'], old.f1, atol=1e-12)
                    rows.append(dict(replicate=rep, scene=index, scorer=kind, decoder=decoder, **metric))
                    saved_labels[f'{kind}_{decoder}'].append(z)
        if rep == 0:
            for kind, original in [('physics','physics_calibrated'),('joint','joint_calibrated'),('boosted','boosted_calibrated')]:
                old = np.load(ROOT/f'results/prediction_{original}_r00.npz')
                probability = np.concatenate(saved_scores[kind])
                probability = np.where(old['gate'], probability, 0.)
                delta = float(np.max(np.abs(old['prob']-probability)))
                assert delta < 1e-8
                agreement.append(dict(scorer=kind, max_probability_difference=delta))
        arrays = {k: np.concatenate(v) for k,v in saved_labels.items()}
        arrays.update({'score_'+k: np.concatenate(v) for k,v in saved_scores.items()})
        arrays.update(observation_boundaries=np.array(boundaries), pair_boundaries=np.array(pair_boundaries))
        np.savez_compressed(OUT/f'predictions_r{rep:02d}.npz', **arrays)
        print(f'Replication {rep+1}/10: {time.time()-started:.1f}s', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'per_scene.csv', index=False)
    replicates = frame.groupby(['replicate','scorer','decoder'], as_index=False)[METRICS].mean()
    replicates.to_csv(OUT/'per_replicate.csv', index=False)
    summary = summarize(replicates, ['scorer','decoder'], METRICS)
    summary.to_csv(OUT/'summary.csv', index=False)
    differences = []
    for kind in KINDS:
        wide = replicates[replicates.scorer == kind].pivot(index='replicate', columns='decoder', values='f1')
        for base in ['components','assignment']:
            for rep in range(10):
                differences.append(dict(replicate=rep, scorer=kind, baseline=base,
                                        delta=float(wide.loc[rep,'constrained']-wide.loc[rep,base])))
    differences = pd.DataFrame(differences)
    differences.to_csv(OUT/'paired_per_replicate.csv', index=False)
    summarize(differences, ['scorer','baseline'], ['delta']).to_csv(OUT/'paired_summary.csv', index=False)
    labels = {'physics':'Physical linear','joint':'Joint linear','unweighted':'Unweighted joint','boosted':'Boosted joint'}
    decoders = {'components':'Components','assignment':'Assignment + union','constrained':'Constrained'}
    lines = [r'\begin{tabularx}{\linewidth}{@{}llYrr@{}}\toprule',
             r'Scorer & Decoder & Event-pair F1 [95\% CI] & Receiver conflict & Gate violation \\\midrule']
    for kind in KINDS:
        for decoder in DECODERS:
            a = summary[(summary.scorer == kind)&(summary.decoder == decoder)].iloc[0]
            lines.append(f"{labels[kind]} & {decoders[decoder]} & {a.f1:.4f} [{a.f1_lo:.4f}, {a.f1_hi:.4f}] & {100*a.receiver_violation:.2f}\\% & {100*a.gate_violation:.2f}\\% "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/factorial.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit = dict(status='PASS', existing_test_scenes=800, scorer_decoder_conditions=12,
                 scene_decodings=len(frame), original_constrained_f1_reproduced=True,
                 replication0_probability_checks=agreement,
                 original_experiment_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Existing synthetic data only; paired decoder comparisons conditional on identical scorer outputs.')
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(summary[['scorer','decoder','f1','receiver_violation','gate_violation']].to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
