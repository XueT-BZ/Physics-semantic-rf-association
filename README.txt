Physics–Semantic Consistency-Constrained Association of Heterogeneous RF Observations for Multi-Receiver Communication Signal Monitoring
Scientific reproducibility project, synchronized 2026-10-29

Contents
  code/                 all 45 supplied Python source files
  data/                 original public sample, metadata and attribution
  results/              controlled descriptor datasets, frozen fits, predictions,
                        per-scene metrics, summaries and experiment protocols
  main.tex, supplement.tex, references.tex, sections/, figures/, tables/
  figure_sources/, Definitions/, main.pdf, supplement.pdf
                        the current 31-page manuscript and 24-page supplement,
                        their editable sources and required document assets
  requirements*.txt     recorded original and subsequent environment versions
  deep_revision_audit/source_manifest.json
  major_review_audit/baseline_manifest.json, protected_inputs.json
  audit/references_2022_2026_review.csv
                        retained records used by verification entry points

Use a separate Python environment for the supplied locked validation environment:
  python -m pip install -r requirements-review-lock.txt

Verify saved independent outputs without fitting new classifiers:
  python code/deep_revision.py --task verify
  python code/review_revision.py --task verify
These commands recompute metrics from retained predictions and write verification
reports under results/. They do not rerun all experiments or fit new classifiers.

Run the independent validation experiments from their supplied inputs:
  python code/deep_revision.py --task all
Rerun the original controlled experiment:
  python code/experiment.py --out results --seeds 10
The original environment is recorded in requirements.txt; subsequent experiments
have separate lock files. Reruns overwrite generated outputs in the extracted
copy, and wall-clock measurements depend on the machine. Retain the saved outputs
for comparison. Document-formatting helpers are retained among the original code;
use the scientific entry points above to reproduce experiments.

The public six-file LoED sample is retained without modification in
data/public/loed/LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip. Its DOI, license,
checksums and attribution are in the same directory. Controlled data contain
measurement descriptors; they are not IQ waveform recordings. The saved-output
verification does not require downloading the public sample again.

The optional parent-workspace protection manifest is checked only for files
present on the current machine; those old absolute paths are not compilation
or experimental input dependencies. Scientific inputs use project-relative paths.

Compile the two documents from this project directory:
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex supplement.tex
The bundled PDFs and document sources match the companion source packages.
All experiment code, input data, fitted parameters, predictions and scientific
results are retained byte-for-byte from the preceding reproducibility package.
SHA256SUMS.txt covers all payload files except itself.
