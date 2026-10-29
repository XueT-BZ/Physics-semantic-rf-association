"""Paired descriptor interventions for semantic association under frozen scorers.

The trace generator replays the released generator exactly. Hidden event families
and copying parents are retained only by the generator and evaluation code.
No waveform or trained RF recognition front end is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
from experiment import Config, Scene, C, K, UNIT_SCALE, simulate, features
from partial_mapping_extension import controlled_adapter
from review_common import metric
from deep_scoring import load_scorers

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/deep_revision/semantic_stress'
CONDITIONS = ['nominal', 'same_class_collisions', 'dense_independent_classes',
              'dense_same_class_collisions', 'imbalanced_classes',
              'receiver_specific_classifiers', 'independent_errors_control',
              'event_shared_errors']
TITLES = {
    'nominal': 'Nominal independent classes',
    'same_class_collisions': 'Same-class near collisions',
    'dense_independent_classes': 'Dense near collisions / independent classes',
    'dense_same_class_collisions': 'Dense near collisions / same class',
    'imbalanced_classes': 'Class prior 0.7/0.1/0.1/0.1',
    'receiver_specific_classifiers': 'Receiver-specific confusion and confidence',
    'independent_errors_control': 'Independent errors / matched marginals',
    'event_shared_errors': 'Event-shared errors / matched marginals',
}
SCORERS = ['physics', 'joint', 'unweighted', 'boosted']


@dataclass
class Trace:
    scene: Scene
    event_classes: np.ndarray
    parents: np.ndarray
    true_class: np.ndarray
    event: np.ndarray
    wrong: np.ndarray
    offset: np.ndarray
    confidence_uniform: np.ndarray
    missing: np.ndarray


def traced_original(seed: int, cfg: Config) -> Trace:
    """Same random draws and order as experiment.simulate; retain latent trace."""
    rng = np.random.default_rng(seed)
    S, E = cfg.receivers, cfg.events
    angles = np.arange(S)*2*np.pi/S + rng.uniform(0, 2*np.pi)
    pos = 2000*np.column_stack([np.cos(angles), np.sin(angles)])
    b = rng.normal(0, cfg.clock_us*1e-6, S)
    d = rng.normal(0, cfg.freq_bias_khz*1e3, S)
    tau = rng.uniform(0, cfg.window_us*1e-6, E)
    carrier = 100e6 + rng.integers(0, 6, E)*1800 + rng.normal(0, 180, E)
    bw = rng.uniform(20e3, 60e3, E)
    dur = rng.uniform(20e-6, 80e-6, E)
    loc = rng.uniform(-1500, 1500, (E, 2))
    protocol = rng.integers(0, K, E)
    parents = np.full(E, -1, int)
    for e in range(1, E):
        if rng.random() < cfg.collisions:
            a = int(rng.integers(0, e)); parents[e] = a
            tau[e] = tau[a] + rng.normal(0, 1.5e-6)
            carrier[e] = carrier[a] + rng.normal(0, 150)
            bw[e] = max(1e3, bw[a]+rng.normal(0, 2e3))
            dur[e] = max(1e-6, dur[a]+rng.normal(0, 2e-6))
    rows, truths, noise_labels, traces = [], [], [], []
    for s in range(S):
        st = (.6+.4*s)*1e-6; sf = (.12+.08*s)*1e3
        sb = (1.5+.3*s)*1e3; sd = (1.0+.3*s)*1e-6
        ids = list(np.flatnonzero(rng.random(E) < cfg.detection))
        false_count = rng.poisson(cfg.clutter)
        for e in ids + [-1]*false_count:
            false = e < 0
            if false:
                tx, fx, wx, lx = (rng.uniform(0, cfg.window_us*1e-6),
                    100e6+rng.integers(0, 6)*1800+rng.normal(0, 180),
                    rng.uniform(20e3, 60e3), rng.uniform(20e-6, 80e-6))
                q = int(rng.integers(K)); delay = 0.
            else:
                tx, fx, wx, lx, q = tau[e], carrier[e], bw[e], dur[e], int(protocol[e])
                delay = np.linalg.norm(loc[e]-pos[s])/C
            jitter = (rng.standard_t(3)/np.sqrt(3) if cfg.heavy_tail else rng.normal())*st
            excess = (rng.exponential(cfg.multipath_mean_us*1e-6)
                      if rng.random() < cfg.multipath_probability else 0.)
            t = tx+delay+b[s]+jitter+excess
            f = fx+d[s]+rng.normal(0, sf)
            w = max(1., wx+rng.normal(0, sb)); length = max(1e-9, lx+rng.normal(0, sd))
            missing = rng.random() < cfg.semantic_missing
            wrong = rng.random() < cfg.semantic_error
            off = int(rng.integers(1, K)) if wrong else 0
            qhat = (q+off) % K
            conf = rng.uniform(.45, .9) if wrong else rng.uniform(.6, .95)
            conf_u = (conf-.45)/.45 if wrong else (conf-.6)/.35
            if cfg.corrupted_receiver and s == S-1:
                qhat = (q+1) % K; conf = .99; missing = False
            probs = np.full(K, (1-conf)/(K-1)); probs[qhat] = conf
            if missing: probs[:] = 1/K
            native = probs[(np.arange(K)+s) % K]
            u = int(rng.integers(3))
            rows.append([s, t, f/UNIT_SCALE[u], u, w, length, st, sf, sb, sd,
                         cfg.declared_clock_us*1e-6, cfg.declared_freq_khz*1e3, *native])
            truths.append(int(e) if not false else E+len(truths))
            noise_labels.append(false)
            traces.append([q, int(e), int(wrong), off, conf_u, int(missing)])
    order = rng.permutation(len(rows))
    a = np.asarray(rows, float)[order]
    v = np.asarray(traces, float)[order]
    names = ['receiver', 'time_s', 'frequency_value', 'frequency_unit', 'bandwidth_hz',
             'duration_s', 'sigma_time_s', 'sigma_frequency_hz', 'sigma_bandwidth_hz',
             'sigma_duration_s', 'declared_clock_s', 'declared_freq_bias_hz']
    obs = {name: a[:, i] for i, name in enumerate(names)}
    obs['native_semantic'] = a[:, 12:16]
    obs['receiver'] = obs['receiver'].astype(int)
    obs['frequency_unit'] = obs['frequency_unit'].astype(int)
    scene = Scene(obs, np.asarray(truths)[order], np.asarray(noise_labels)[order], pos, seed)
    return Trace(scene, protocol.copy(), parents, v[:, 0].astype(int), v[:, 1].astype(int),
                 v[:, 2].astype(bool), v[:, 3].astype(int), v[:, 4], v[:, 5].astype(bool))


def confusion_matrices():
    """Equal 0.12 marginal errors; source-specific off-diagonal structure."""
    matrices = np.zeros((4, 4, 4))
    for s in range(4):
        np.fill_diagonal(matrices[s], .88)
        for q in range(4):
            if s == 0:
                for a in range(4):
                    if a != q: matrices[s, q, a] = .04
            elif s == 1: matrices[s, q, (q+1) % 4] = .12
            elif s == 2: matrices[s, q, q ^ 1] = .12
            else: matrices[s, q, (q-1) % 4] = .12
    return matrices


def intervene(trace: Trace, condition: str):
    """Return observable scene and separate generation/evaluation metadata."""
    base = trace.scene
    scene = Scene({k: v.copy() for k, v in base.obs.items()}, base.truth.copy(),
                  base.clutter.copy(), base.positions.copy(), base.seed)
    q = trace.true_class.copy(); event_q = trace.event_classes.copy()
    wrong = trace.wrong.copy(); off = trace.offset.copy(); cu = trace.confidence_uniform.copy()
    missing = trace.missing.copy(); receiver = scene.obs['receiver']; n = len(q)
    real = trace.event >= 0
    label_rng = np.random.default_rng(np.random.SeedSequence([base.seed, 8021]))
    if condition in ['same_class_collisions', 'dense_same_class_collisions']:
        for e, parent in enumerate(trace.parents):
            if parent >= 0: event_q[e] = event_q[parent]
        q[real] = event_q[trace.event[real]]
    if condition == 'imbalanced_classes':
        event_q = label_rng.choice(4, size=len(event_q), p=[.7, .1, .1, .1])
        q[real] = event_q[trace.event[real]]
        q[~real] = label_rng.choice(4, size=int((~real).sum()), p=[.7, .1, .1, .1])
    qhat = (q+off) % 4
    conf = np.where(wrong, .45+.45*cu, .60+.35*cu)
    if condition == 'receiver_specific_classifiers':
        rng = np.random.default_rng(np.random.SeedSequence([base.seed, 8022]))
        matrices = confusion_matrices(); draw = rng.random(n)
        qhat = np.sum(draw[:, None] > np.cumsum(matrices[receiver, q], axis=1), axis=1)
        wrong = qhat != q
        ranges = np.asarray([[.60, .95], [.70, .98], [.45, .80], [.90, .99]])
        conf = ranges[receiver, 0]+cu*(ranges[receiver, 1]-ranges[receiver, 0])
    if condition in ['independent_errors_control', 'event_shared_errors']:
        rng = np.random.default_rng(np.random.SeedSequence([base.seed, 8023]))
        # Fixed report draws serve both conditions and keep clutter identical.
        wrong = rng.random(n) < .12; off = rng.integers(1, 4, n)
        cu = rng.random(n)
        event_wrong = rng.random(len(event_q)) < .12
        event_off = rng.integers(1, 4, len(event_q))
        if condition == 'event_shared_errors':
            wrong[real] = event_wrong[trace.event[real]]
            off[real] = event_off[trace.event[real]]
        qhat = np.where(wrong, (q+off) % 4, q)
        conf = np.where(wrong, .45+.45*cu, .60+.35*cu)
    probs = np.repeat(((1-conf)/3)[:, None], 4, axis=1)
    probs[np.arange(n), qhat] = conf
    probs[missing] = .25
    # Preserve original native vectors exactly for unchanged nominal references.
    if condition not in ['nominal', 'dense_independent_classes']:
        scene.obs['native_semantic'] = probs[(np.arange(n)[:, None],
                                              (np.arange(4)[None, :]+receiver[:, None]) % 4)]
    else:
        for i, s in enumerate(receiver):
            probs[i, (np.arange(4)+s) % 4] = scene.obs['native_semantic'][i]
    assert all(np.array_equal(scene.obs[k], base.obs[k])
               for k in scene.obs if k != 'native_semantic')
    assert np.allclose(probs.sum(axis=1), 1.)
    parents = np.flatnonzero(trace.parents >= 0)
    metadata = dict(true_class=q, predicted_class=qhat, missing=missing, confidence=conf,
                    wrong=wrong, event=trace.event.copy(), event_classes=event_q,
                    collision_events=parents,
                    same_class_collisions=int(np.sum(event_q[parents] == event_q[trace.parents[parents]])))
    return scene, metadata


def make_scene(seed: int, condition: str):
    if condition not in CONDITIONS: raise ValueError(condition)
    cfg = replace(Config(), collisions=.65) if condition.startswith('dense_') else Config()
    return intervene(traced_original(seed, cfg), condition)


def equality(a, b):
    return (all(np.array_equal(a.obs[k], b.obs[k]) for k in a.obs)
            and np.array_equal(a.truth, b.truth) and np.array_equal(a.clutter, b.clutter)
            and np.array_equal(a.positions, b.positions))


def aggregate(frame, group_columns, columns):
    output = []
    for keys, g in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_columns, keys))
        row['replications'] = len(g)
        for c in columns:
            values = g[c].to_numpy(float)
            m = float(values.mean()); h = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
            row.update({c: m, c+'_lo': m-h, c+'_hi': m+h})
        output.append(row)
    return pd.DataFrame(output)


def artifacts(summary):
    by = summary.set_index(['condition', 'scorer'])
    lines = [r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule',
             r'Condition & Physical & Weighted linear & Unweighted linear & Boosted \\\midrule']
    for condition in CONDITIONS:
        cells = [TITLES[condition]]
        for scorer in SCORERS:
            r = by.loc[(condition, scorer)]
            cells.append(f"{r.f1:.4f}")
        lines.append(' & '.join(cells)+r' \\')
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/deep_semantic_stress.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Full intervals stay in an additional narrow table suitable for the supplement.
    lines = [r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule',
             r'Condition & Physical F1 & Weighted F1 & Unweighted F1 & Boosted F1 \\\midrule']
    for condition in CONDITIONS:
        cells = [TITLES[condition]]
        for scorer in SCORERS:
            r = by.loc[(condition, scorer)]
            cells.append(f"{r.f1:.4f} [{r.f1_lo:.4f}, {r.f1_hi:.4f}]")
        lines.append(' & '.join(cells)+r' \\')
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/deep_semantic_intervals.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(scenes=40):
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = dict(replications=10, scenes_per_replication=scenes,
        conditions=CONDITIONS, scorer_names=SCORERS,
        test_seed='110000000 + 100000*r + j; r=0..9, j=0..39',
        class_vocabulary=4, receivers=4, hidden_events=14,
        default_collision_probability=.25, dense_collision_probability=.65,
        nominal_class_prior=[.25]*4, imbalanced_class_prior=[.7,.1,.1,.1],
        same_class_rule='Copy the parent event class in event index order; independent clutter classes are retained.',
        classifier_matrices=confusion_matrices().tolist(),
        receiver_confidence_ranges=[[.60,.95],[.70,.98],[.45,.80],[.90,.99]],
        receiver_specific_confidence_depends_on_correctness=False,
        shared_error_probability=.12,
        shared_error_rule='A single wrong flag and nonzero cyclic class offset are drawn per genuine event and shared by its receivers. Per-report confidence draws and missingness remain independent. Clutter uses independent report-level errors.',
        independent_error_control='Same .12 error probability, uniform nonzero offset, confidence laws and missingness; error flags and offsets are drawn per report.',
        confidence_if_correct=[.60,.95], confidence_if_wrong=[.45,.90],
        missing_semantics_probability=.10,
        inference_maps='All maps supplied correctly; this audit isolates evidence quality. Unknown maps are evaluated separately.',
        fits='Frozen original linear parameters; original-hyperparameter boosted refits reconstructed from the original train/calibration splits and cached by deep_scoring.py.',
        new_test_fitting=False, output_metric='Scene-macro event-pair F1; 95% t intervals over ten replication means',
        scope='Descriptor-level stress test, not measured classifier or waveform validation',
        original_generator_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DEST/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.perf_counter(); rows=[]; audits=[]; observations=[]; labels={}; equality_count=0
    expected_maps = np.tile(np.arange(4), (4,1))
    for rep in range(10):
        models = load_scorers(rep)
        for j in range(scenes):
            seed = 110_000_000+100_000*rep+j
            traces = {False: traced_original(seed, Config()),
                      True: traced_original(seed, replace(Config(), collisions=.65))}
            for dense, trace in traces.items():
                assert equality(trace.scene, simulate(seed, replace(Config(), collisions=.65) if dense else Config()))
                equality_count += 1
            for condition in CONDITIONS:
                trace = traces[condition.startswith('dense_')]
                scene, m = intervene(trace, condition)
                b, p = controlled_adapter(scene, expected_maps)
                if j == 0:
                    altered = Scene(scene.obs, np.roll(scene.truth, 1), ~scene.clutter, scene.positions, scene.seed)
                    bb, _ = controlled_adapter(altered, expected_maps)
                    assert all(np.array_equal(getattr(b, name), getattr(bb, name))
                               for name in ['receiver','native','i','j','physical','time_z','freq_z'])
                count = len(m['collision_events'])
                audits.append(dict(replication=rep, scene=j, seed=seed, condition=condition,
                    reports=len(scene.truth), collision_pairs=count,
                    same_class_collision_pairs=m['same_class_collisions'],
                    **{f'event_class_{q}':int(np.sum(m['event_classes']==q)) for q in range(4)}))
                for i in range(len(scene.truth)):
                    observations.append(dict(replication=rep, scene=j, condition=condition,
                        row=i, receiver=int(b.receiver[i]), event=int(m['event'][i]),
                        true_class=int(m['true_class'][i]), predicted_class=int(m['predicted_class'][i]),
                        missing=bool(m['missing'][i]), confidence=float(m['confidence'][i]),
                        wrong=bool(m['wrong'][i])))
                for scorer in SCORERS:
                    z = models[scorer].decode(b, expected_maps)
                    result = metric(scene, p, z)
                    assert result['receiver_conflicts']==result['gate_violations']==0
                    rows.append(dict(replication=rep, scene=j, seed=seed, condition=condition,
                                     scorer=scorer, reports=len(z), **result))
                    labels[f'r{rep}_{j}_{condition}_{scorer}'] = z
        print(f'Semantic stress {rep+1}/10; {len(rows)} outcomes; {time.perf_counter()-started:.1f}s', flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'per_scene.csv', index=False)
    np.savez_compressed(DEST/'predictions.npz', **labels)
    checks = pd.DataFrame(audits); checks.to_csv(DEST/'generation_audit.csv', index=False)
    report_audit = pd.DataFrame(observations); report_audit.to_csv(DEST/'classification_audit.csv.gz', index=False)
    metrics = ['f1','precision','recall','ari','b3f1','exact','clutter_attachment',
               'true_pair_gate_recall','receiver_violation','gate_violation']
    per_rep = frame.groupby(['condition','scorer','replication'], sort=False)[metrics].mean().reset_index()
    per_rep.to_csv(DEST/'per_replication.csv', index=False)
    summary = aggregate(per_rep, ['condition','scorer'], metrics)
    summary.to_csv(DEST/'summary.csv', index=False)
    counts = frame.groupby(['condition','scorer'])[['tp','fp','fn','reports']].sum().reset_index()
    counts['micro_f1'] = 2*counts.tp/(2*counts.tp+counts.fp+counts.fn)
    counts.to_csv(DEST/'pooled_counts.csv', index=False)
    paired = []
    contrasts = [('same_class_collisions','nominal'),
                 ('dense_same_class_collisions','dense_independent_classes'),
                 ('event_shared_errors','independent_errors_control')]
    ix = per_rep.set_index(['condition','scorer','replication'])
    for target, reference in contrasts:
        for scorer in SCORERS:
            for rep in range(10):
                paired.append(dict(target=target, reference=reference, scorer=scorer, replication=rep,
                    f1=float(ix.loc[(target,scorer,rep),'f1']-ix.loc[(reference,scorer,rep),'f1'])))
    for condition in CONDITIONS:
        for scorer in ['joint','unweighted','boosted']:
            for rep in range(10):
                paired.append(dict(target=condition, reference='physics', scorer=scorer, replication=rep,
                    f1=float(ix.loc[(condition,scorer,rep),'f1']-ix.loc[(condition,'physics',rep),'f1'])))
    contrast_frame = pd.DataFrame(paired)
    contrast_frame.to_csv(DEST/'paired_differences_per_replication.csv', index=False)
    aggregate(contrast_frame, ['target','reference','scorer'], ['f1']).to_csv(DEST/'paired_differences.csv', index=False)
    # Every semantic-only intervention must leave the physical-only result unchanged.
    base = frame[frame.scorer=='physics'].set_index(['condition','replication','scene'])
    for condition in CONDITIONS:
        reference = 'dense_independent_classes' if condition.startswith('dense_') else 'nominal'
        for rep in range(10):
            for j in range(scenes):
                assert base.loc[(condition,rep,j),'f1']==base.loc[(reference,rep,j),'f1']
    empirical = []
    for (condition,s), g in report_audit[~report_audit.missing].groupby(['condition','receiver']):
        genuine = g[g.event >= 0]
        empirical.append(dict(condition=condition, receiver=int(s), reports=len(genuine),
            error_fraction=float(genuine.wrong.mean()), mean_confidence=float(genuine.confidence.mean()),
            wrong_mean_confidence=float(genuine.loc[genuine.wrong,'confidence'].mean())))
    pd.DataFrame(empirical).to_csv(DEST/'empirical_classifiers.csv', index=False)
    report_audit[(~report_audit.missing)&(report_audit.event>=0)].groupby(
        ['condition','receiver','true_class','predicted_class']).size().rename('count').to_csv(DEST/'empirical_confusion_counts.csv')
    original = checks[checks.condition=='nominal']
    original_same = int(original.same_class_collision_pairs.sum()); original_count = int(original.collision_pairs.sum())
    same = checks[checks.condition.isin(['same_class_collisions','dense_same_class_collisions'])]
    assert np.array_equal(same.same_class_collision_pairs.to_numpy(), same.collision_pairs.to_numpy())
    artifacts(summary)
    verification = dict(status='PASS', elapsed_seconds=time.perf_counter()-started,
        scene_outcomes=len(frame), traced_original_equalities=equality_count,
        physical_arrays_unchanged_in_semantic_only_conditions=True,
        physical_only_results_identical_for_paired_semantic_conditions=True,
        all_receiver_and_gate_violations_zero=True,
        nominal_same_class_collision_pairs=original_same, nominal_collision_pairs=original_count,
        nominal_observed_same_class_fraction=original_same/original_count,
        nominal_expected_same_class_fraction=.25,
        inference_features_unchanged_when_truth_and_clutter_flags_altered=True,
        separate_independent_test_seeds=True, python=sys.version, platform=platform.platform())
    (DEST/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
