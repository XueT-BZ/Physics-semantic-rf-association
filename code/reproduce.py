#!/usr/bin/env python3
"""Reproduce the delivered study or rebuild its already-executed results.

  python code/reproduce.py                     # full 10-replication rerun
  python code/reproduce.py --skip-experiments  # redraw, compile, and audit only

This script overwrites generated results/figures/PDFs in this project. Keep a copy
of the delivered archive before comparing fresh timing measurements.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'audit'

def execute(args: list[str], log_name: str) -> None:
    print('Running:', ' '.join(map(str,args)), flush=True)
    with (AUDIT/log_name).open('w',encoding='utf-8') as log:
        completed=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True)
    if completed.returncode:
        tail=(AUDIT/log_name).read_text(encoding='utf-8',errors='replace')[-4000:]
        raise RuntimeError(f'Command failed; see audit/{log_name}\n{tail}')

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-experiments',action='store_true',help='Use supplied result tables instead of refitting.')
    parser.add_argument('--publication-experiments',action='store_true',help='Run measured LoED, crossed scorer-decoder and semantic-mapping experiments using the supplied inputs.')
    args=parser.parse_args()
    if sys.version_info < (3,11):
        raise RuntimeError('Python 3.11+ is required; see requirements*.txt for the recorded original and publication environments.')
    if not shutil.which('pdflatex'):
        raise RuntimeError('pdflatex is not on PATH. Install a TeX distribution with TikZ and standalone first.')
    AUDIT.mkdir(exist_ok=True)
    if not args.skip_experiments:
        execute([sys.executable,'code/experiment.py','--out',str(ROOT/'results'),'--seeds','10'],
                'fresh_experiment.log')
    if args.publication_experiments:
        execute([sys.executable,'code/scorer_decoder_factorial.py'],'publication_factorial.log')
        execute([sys.executable,'code/loed_benchmark.py'],'publication_loed.log')
        execute([sys.executable,'code/semantic_alignment_audit.py'],'semantic_alignment.log')
    if (ROOT/'results/loed/summary.csv').exists():
        execute([sys.executable,'code/format_publication_tables.py'],'format_publication_tables.log')
    for filename in ['audit_event_definitions.py','make_diagrams.py','make_results_figures.py','check_curve_integrity.py','make_figure_atlas.py','make_comparison_atlas.py','test_invariants.py','verify_diagram_semantics.py']:
        execute([sys.executable,'code/'+filename],filename.removesuffix('.py')+'.log')
    for document in ['main','supplement']:
        if shutil.which('latexmk'):
            execute(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',document+'.tex'],
                    'build_'+document+'.log')
        else:
            for iteration in range(3):
                execute(['pdflatex','-interaction=nonstopmode','-halt-on-error',document+'.tex'],
                        f'build_{document}_{iteration+1}.log')
    execute([sys.executable,'code/verify_results.py'],'verification_run.log')
    if (ROOT/'results/loed/audit.json').exists():
        execute([sys.executable,'code/verify_publication_results.py'],'verify_publication_results.log')
    if (ROOT/'results/semantic_alignment/audit.json').exists():
        execute([sys.executable,'code/verify_semantic_alignment.py'],'verify_semantic_alignment.log')
    execute([sys.executable,'code/document_quality.py'],'document_quality_run.log')
    print('Reproduction and audits completed. Inspect main.pdf, supplement.pdf, and audit/*.json.')
    print('Scientific outputs are deterministic under the pinned environment; runtimes are not.')

if __name__=='__main__':
    main()
