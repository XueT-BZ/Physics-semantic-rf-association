#!/usr/bin/env python3
"""Run the paper's experiments, rebuild saved artifacts, or verify delivered results.

  python code/reproduce.py --verify-only
  python code/reproduce.py --skip-experiments --skip-latex
  python code/reproduce.py --all-experiments --skip-latex

The default reruns the original ten-replication experiment and rebuilds artifacts.
--all-experiments also executes the public, missing-map, stress and latency studies.
Keep the delivered results for comparison before requesting a fresh experiment run.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'audit/reproduce'


def execute(script: str, *arguments: str) -> None:
    args = [sys.executable, str(ROOT / 'code' / script), *arguments]
    command(args, script.removesuffix('.py') + '.log')


def command(args: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    print('Running:', ' '.join(map(str, args)), flush=True)
    target = LOGS / log_name
    with target.open('w', encoding='utf-8') as log:
        completed = subprocess.run(args, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        tail = target.read_text(encoding='utf-8', errors='replace')[-4000:]
        raise RuntimeError(f'Command failed; see {target.relative_to(ROOT)}\n{tail}')


def verify() -> None:
    # Each check uses delivered predictions or small mathematical examples.
    # No classifier fitting, dataset download or TeX installation is required.
    for script in ['test_invariants.py', 'verify_diagram_semantics.py',
                   'verify_results.py', 'verify_publication_results.py',
                   'verify_semantic_alignment.py', 'verify_partial_mapping_extension.py',
                   'verify_review_revision.py', 'verify_detail_revision.py',
                   'check_curve_integrity.py']:
        execute(script)
    execute('deep_revision.py', '--task', 'verify')
    print('All numerical verification checks passed. Reports and logs are under audit/.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true', help='Verify delivered outputs without refitting, downloading or compiling.')
    mode.add_argument('--skip-experiments', action='store_true', help='Rebuild figures and derived tables from supplied results.')
    mode.add_argument('--all-experiments', action='store_true', help='Rerun every experimental study represented in the paper and supplement.')
    parser.add_argument('--publication-experiments', action='store_true', help='Also run the public, scorer-decoder and original alignment/missing-map studies.')
    parser.add_argument('--skip-latex', action='store_true', help='Skip TikZ diagram and document compilation; numerical work needs no TeX.')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11+ is required; use the supplied recorded dependency locks.')
    if args.verify_only:
        if args.publication_experiments:
            parser.error('--verify-only cannot be combined with --publication-experiments')
        verify()
        return
    if not args.skip_latex and not shutil.which('pdflatex'):
        parser.error('Install a TeX distribution with TikZ or add --skip-latex.')
    LOGS.mkdir(parents=True, exist_ok=True)
    if not args.skip_experiments:
        execute('experiment.py', '--out', str(ROOT / 'results'), '--seeds', '10')
    if args.publication_experiments or args.all_experiments:
        for script in ['loed_benchmark.py', 'scorer_decoder_factorial.py',
                       'semantic_alignment_audit.py', 'partial_mapping_extension.py']:
            execute(script)
    if args.all_experiments:
        execute('review_revision.py', '--task', 'all')
        execute('deep_revision.py', '--task', 'all')
    for script in ['audit_event_definitions.py', 'make_results_figures.py',
                   'format_publication_tables.py', 'make_review_artifacts.py']:
        execute(script)
    execute('deep_mapping.py', '--task', 'artifacts')
    execute('deep_semantic_stress.py', '--artifacts-only')
    execute('deep_semantic_artifacts.py')
    execute('deep_runtime.py', '--table-only')
    if not args.skip_latex:
        execute('make_diagrams.py')
        build = ROOT / 'build'
        build.mkdir(exist_ok=True)
        for name in ['main', 'supplement']:
            if shutil.which('latexmk'):
                command(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                         '-outdir=build', name + '.tex'], 'build_' + name + '.log')
            else:
                for iteration in range(3):
                    command(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                             '-output-directory=build', name + '.tex'],
                            f'build_{name}_{iteration + 1}.log')
            shutil.copy2(build / (name + '.pdf'), ROOT / (name + '.pdf'))
    verify()
    print('Reproduction completed. Wall-clock timings depend on the execution machine.', flush=True)


if __name__ == '__main__':
    main()
