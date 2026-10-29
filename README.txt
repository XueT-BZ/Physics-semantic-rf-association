Physics–Semantic RF Observation Association

Code, data and reproducibility materials for the manuscript Physics–Semantic Consistency-Constrained Association of Heterogeneous RF Observations for Multi-Receiver Communication Signal Monitoring, by Yuanfeng Gao, Wenbo Xu and Bojun Zhao.

Repository: https://github.com/XueT-BZ/Physics-semantic-rf-association

The study associates reports from heterogeneous RF receivers using physical admission, class correspondence and constrained event grouping. Class-probability experiments use controlled descriptors. The public LoED check evaluates reception metadata and packet references. The delivered data are not IQ waveform recordings.

Contents

| Location | Contents |
| --- | --- |
| code/ | 36 experimental, figure/table and numerical-verification modules |
| CODE_INDEX.csv | Purpose, supplementary section and imports for every retained module |
| data/public/loed/ | Original public sample ZIP, metadata, attribution and license |
| results/ | Controlled descriptors, frozen parameters, predictions and reported results |
| figures/, figure_sources/, tables/ | Published figure/table assets and editable diagram sources |
| main.pdf, supplement.pdf | Manuscript and separate online supplementary material |
| main.tex, supplement.tex, sections/, references.tex, Definitions/ | Complete document sources and required local template assets |
| requirements*.txt | Recorded numerical environments for the original and subsequent studies |
| SHA256SUMS.txt | Checksums of the distributed payload, excluding this checksum file itself |

Environment

The independent validation was executed with Python 3.12.14 and the packages in requirements-review-lock.txt:

python -m pip install -r requirements-review-lock.txt

The other requirements files preserve the environments used for earlier experiment stages. TeX is needed only to regenerate TikZ diagrams or compile the manuscript. The numerical checks below require neither TeX nor network access.

Verify the delivered results

Run from the repository root:

python code/reproduce.py --verify-only

This checks unit/vocabulary invariants; all 2400 nominal scenes; saved public, alignment and missing-map predictions; conditional-search examples; scorer/decoder counts; plotted differences; and the independent mapping, semantic and runtime partitions. It does not fit classifiers or download data. Logs and reports are written under audit/; the coordinate-trap check also refreshes its small mathematical records under results/detail_revision/.

Individual entry points remain available:

python code/verify_results.py
python code/verify_partial_mapping_extension.py
python code/review_revision.py --task verify
python code/review_revision.py --task detail
python code/deep_revision.py --task verify

Rebuild figures and derived tables

Use the saved outputs without rerunning the experiments:

python code/reproduce.py --skip-experiments --skip-latex

This rebuilds result plots and derived tables, including the original missing-map table, and then runs the numerical checks. It does not rewrite manuscript paragraphs. To regenerate TikZ diagrams and compile both documents as well, omit --skip-latex; install a TeX distribution with pdfLaTeX, TikZ and standalone. latexmk is used when available.

Rerun the experiments

Use a fresh copy to retain the supplied outputs for comparison. A complete rerun overwrites experimental outputs, can take substantially longer than verification, and records new hardware-dependent timings:

python code/reproduce.py --all-experiments --skip-latex

The stages can also be executed separately:

| Study | Command | Results |
| --- | --- | --- |
| Original ten-replication benchmark | python code/experiment.py --out results --seeds 10 | results/ |
| Measured LoED association | python code/loed_benchmark.py | results/loed/ |
| Frozen scorer/decoder comparison | python code/scorer_decoder_factorial.py | results/factorial/ |
| Known-map and representation controls | python code/semantic_alignment_audit.py | results/semantic_alignment/ |
| Original missing-map probe | python code/partial_mapping_extension.py | results/partial_mapping_extension/ |
| Independent mapping, risk, exact-decoder and public-sensitivity studies | python code/review_revision.py --task all | results/review_revision/, results/detail_revision/ |
| Independent map-estimator, semantic, public-clock and latency studies | python code/deep_revision.py --task all | results/deep_revision/ |

The default python code/reproduce.py reruns the original nominal benchmark and rebuilds saved artifacts. Add --all-experiments to regenerate the complete set of later studies. aggregate.py supports combining separately executed nominal replications; it does not synthesize missing runs.

Public data and interpretation

Keep data/public/loed/LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip intact: the loader reads that ZIP directly. The original six-file sample is redistributed with its attribution and CC BY 4.0 license. Provenance is in the same directory; its source is https://doi.org/10.5281/zenodo.4121430. If the supplied archive is missing, python code/fetch_loed.py downloads it and verifies the publisher checksum.

Descriptor truth, class-map interventions and public packet references have different meanings. Mapping recovery and receiver/gate feasibility do not establish event identity in every condition. The paper and supplement state the relevant collision, calibration and public-reference limitations.

Compile the documents separately

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex supplement.tex

The original manuscript and supplementary PDFs have 31 and 24 pages. The document sources do not depend on any parent workspace, historical trial directory or old revision manifest. Numerical verification is independent of manuscript wording, bibliography count and page count.
