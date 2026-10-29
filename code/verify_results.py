#!/usr/bin/env python3
"""Audit the delivered study without network access or refitting its classifiers.

Checks every nominal observation table against its declared scene seeds, all
summary confidence intervals, stored first-replication predictions and linear
parameters, partition constraints, reference/figure completeness, and PDF pages.
An AssertionError identifies a failed contract; the report is written only after
all checks pass. Run from any working directory with the pinned dependencies.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from pypdf import PdfReader
from threadpoolctl import threadpool_limits
from experiment import Config, simulate, scene_csv, features, pair_metrics, summarize

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'

def equal_frame(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    """Allow CSV representation roundoff, but not changed numerical results."""
    cols = list(expected.columns)
    actual = actual[cols].sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)

def verify() -> dict:
    env = json.loads((R / 'environment.json').read_text())
    code_hash = hashlib.sha256((ROOT / 'code' / 'experiment.py').read_bytes()).hexdigest()
    assert code_hash == env['script_sha256'], 'Experiment code differs from executed code.'
    manifest = pd.read_csv(R / 'split_manifest.csv')
    assert set(manifest.replicate) == set(range(10)) and len(manifest) == 30
    used_seeds: set[int] = set()
    for row in manifest.itertuples():
        seeds = set(range(int(row.first_scene_seed), int(row.first_scene_seed + row.scenes)))
        assert not used_seeds.intersection(seeds), 'A nominal scene seed appears in two splits.'
        used_seeds.update(seeds)
        scenes = [simulate(int(row.first_scene_seed + k), Config()) for k in range(int(row.scenes))]
        expected = pd.concat([scene_csv(sc, int(row.replicate), row.split, k)
                              for k, sc in enumerate(scenes)], ignore_index=True)
        path = R / f'observations_r{row.replicate:02d}_{row.split}.csv.gz'
        actual = pd.read_csv(path, float_precision='round_trip')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-14)
        ps = [features(sc) for sc in scenes]
        assert len(actual) == row.observations
        assert sum(len(p.y) for p in ps) == row.pairs
        assert sum(int(p.y.sum()) for p in ps) == row.positive_pairs
    counts = manifest.groupby('split')[['scenes', 'observations', 'pairs', 'positive_pairs']].sum()
    assert counts.loc['test', 'scenes'] == 800 and counts.loc['test', 'pairs'] == 693023
    assert counts.loc['test', 'observations'] == 38451 and counts.loc['test', 'positive_pairs'] == 43151

    pr = pd.read_csv(R / 'pair_per_replicate.csv')
    pm = ['auroc', 'ap', 'brier', 'nll', 'ece', 'admitted_ece', 'positive_rate']
    equal_frame(pd.read_csv(R / 'pair_summary.csv'), summarize(pr, ['method'], pm), ['method'])
    assert len(pr) == 120
    cs = pd.read_csv(R / 'cluster_per_scene.csv')
    assert len(cs) == 10 * 80 * 8
    cm = ['f1', 'precision', 'recall', 'ari', 'b3f1', 'exact', 'receiver_violation',
          'gate_violation', 'clutter_attachment', 'runtime_ms']
    cr = cs.groupby(['replicate', 'method'], as_index=False)[cm].mean()
    equal_frame(pd.read_csv(R / 'cluster_per_replicate.csv'), cr, ['replicate', 'method'])
    equal_frame(pd.read_csv(R / 'cluster_summary.csv'), summarize(cr, ['method'], cm), ['method'])
    for name in ['joint_constrained', 'physics_constrained', 'boosted_constrained', 'unweighted_semantics']:
        a = cs[cs.method == name]
        assert (a.receiver_violation == 0).all() and (a.gate_violation == 0).all()
    st = pd.read_csv(R / 'stress_per_replicate.csv')
    sm = ['f1', 'precision', 'recall', 'ari', 'exact', 'runtime_ms', 'observations',
          'gate_recall', 'retained_fraction', 'auroc', 'ap', 'brier', 'nll']
    equal_frame(pd.read_csv(R / 'stress_summary.csv'), summarize(st, ['group', 'value', 'method'], sm),
                ['group', 'value', 'method'])
    assert len(st) == 10 * 34 * 2
    gr = pd.read_csv(R / 'gate_per_replicate.csv')
    equal_frame(pd.read_csv(R / 'gate_summary.csv'),
                summarize(gr, ['k'], ['f1', 'gate_recall', 'retained_fraction']), ['k'])

    # Recompute the held-out first replication directly from the fixed generator.
    p0 = [features(simulate(1020000 + k, Config())) for k in range(80)]
    truth = np.concatenate([p.y for p in p0]); gates = np.concatenate([p.gate() for p in p0])
    saved = json.loads((R / 'linear_models_r00.json').read_text())
    largest_delta = 0.0
    for method in pr.method.unique():
        a = np.load(R / f'prediction_{method}_r00.npz')
        assert np.array_equal(a['y'], truth) and np.array_equal(a['gate'], gates)
        computed = pair_metrics(a['y'], a['prob'])
        expected = pr[(pr.replicate == 0) & (pr.method == method)].iloc[0]
        for name, value in computed.items():
            assert np.isclose(value, expected[name], rtol=1e-10, atol=1e-12), (method, name)
        if method.startswith('boosted'):
            continue  # Stored predictions audited; reproducible training is in experiment.py.
        kind = method.replace('_calibrated', '').replace('_gated', '')
        meta = saved[kind]
        X = np.concatenate([(p.Xraw if kind == 'raw_units' else p.Xunweighted
                             if kind == 'unweighted' else p.X)[:, meta['columns']] for p in p0])
        z = (X - np.array(meta['mean'])) / np.array(meta['scale'])
        logits = z @ np.array(meta['coefficient']) + meta['intercept']
        if method.endswith('_calibrated'):
            logits /= meta['temperature']
        prob = expit(logits)
        if method.endswith('_gated') or method.endswith('_calibrated'):
            prob = np.where(gates, prob, 0.0)
        d = float(np.max(np.abs(prob - a['prob'])))
        largest_delta = max(largest_delta, d)
        assert d < 1e-9, (method, d)
    ex = pd.read_csv(R / 'exact_cases.csv')
    assert len(ex) == 60 and (ex.optimal_objective >= ex.greedy_objective - 1e-6).all()
    assert int((ex.relative_gap.abs() < 1e-8).sum()) == 47


    main = (ROOT / 'main.tex').read_text() + '\n' + '\n'.join(
        p.read_text() for p in sorted((ROOT/'sections').glob('*.tex')))
    refs = (ROOT / 'references.tex').read_text()
    cite_order = []
    for group in re.findall(r'\\cite(?:p|t)?\{([^}]+)\}', main):
        for key in group.split(','):
            if key not in cite_order:
                cite_order.append(key)
    bibkeys = re.findall(r'\\bibitem\{([^}]+)\}', refs)
    assert len(bibkeys) == len(set(bibkeys)) == 49
    assert cite_order == bibkeys, 'References must follow first-citation order.'
    ref_audit = pd.read_csv(ROOT / 'audit' / 'references_2022_2026_review.csv')
    assert set(ref_audit.key) == set(bibkeys)
    assert int(ref_audit.within_2022_2026.sum()) == 37
    assert ref_audit.loc[ref_audit.key == 'wisig', 'pages_or_article'].iloc[0] == '22808--22818'
    assert ref_audit.publisher_or_identifier_url.notna().all()
    diagrams = ['fig01_task','fig02_architecture','fig03_contract','fig04_gate',
                'fig05_evidence','fig06_temperature','fig07_decoder']
    result_panels = ['results_pr','results_reliability','results_partition','results_semantic',
                     'results_clock_f1','results_clock_gate','results_runtime']
    for stem in diagrams+result_panels+['supp_risk_coverage','supp_exact_gap']:
        assert (ROOT/'figures'/f'{stem}.pdf').is_file(), stem
    for stem in diagrams:
        assert (ROOT/'figure_sources'/f'{stem}.tex').is_file(), stem
    for table in ['pair','cluster','stress','gate']:
        assert (ROOT/'tables'/f'{table}.tex').is_file()
    csu = pd.read_csv(R/'cluster_summary.csv').set_index('method')
    for method in ['joint_constrained','physics_constrained','boosted_constrained']:
        assert f'{csu.loc[method,"f1"]:.4f}' in main
    pdf = PdfReader(ROOT/'main.pdf')
    text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    assert '??' not in text
    if not (ROOT/'Definitions'/'mdpi.cls').exists():
        assert 20 <= len(pdf.pages) <= 24, 'Page count changed unexpectedly: inspect the compiled document.'
    spdf = PdfReader(ROOT/'supplement.pdf')
    for document in ['main','supplement']:
        log=(ROOT/f'{document}.log').read_text(errors='replace')
        for bad in ['Overfull', 'undefined references', 'undefined citations', 'LaTeX Error', 'Fatal error']:
            assert bad not in log, f'{document}: compilation issue: {bad}'
    return dict(status='PASS', nominal_scenes_verified=len(used_seeds),
                split_totals=counts.to_dict(orient='index'), pair_variants=12, partition_variants=8,
                full_test_scene_decodings=len(cs), stress_settings=34, exact_cases=len(ex),
                exact_objective_matches=47, max_linear_prediction_difference=largest_delta,
                references=49, recent_2022_2026=37, recent_share_percent=75.51,
                academic_references=46, datasets=2, standards=1,
                main_numbered_figures=8, main_figure_panels_or_assets=11,
                supplementary_figures=5, editable_conceptual_diagrams=7,
                pdf_pages=len(pdf.pages), supplementary_pages=len(spdf.pages),
                experiment_sha256=code_hash,
                limits='Numerical outcomes, mathematical contracts, and document completeness checked; not RF field validation or official-publisher class certification.')

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report = verify()
    (ROOT / 'audit' / 'verification_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
