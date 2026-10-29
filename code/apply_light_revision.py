"""Apply a small additive revision inside this independently copied project."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
assert ROOT.name.startswith('Backup_Trial_PartialMapping_')

def replace(file,old,new):
    path=ROOT/file;text=path.read_text(encoding='utf-8')
    assert text.count(old)==1,(file,old)
    path.write_text(text.replace(old,new),encoding='utf-8')

METHOD=r'''\subsection{Optional completion of a partially known class registry}\label{sec:partial_registry}
The preceding pipeline and its diagrams assume a complete registry. A small extension allows selected class entries at one receiver to be unknown, while preserving the original evidence in Equations~\eqref{eq:sem}--\eqref{eq:entropy}, fitted coefficients, physical gate and Algorithm~\ref{alg:greedy}. Other receiver maps, field definitions and units remain known. The shared class set and its size are fixed, anchors are correct, and the incomplete map remains a bijection across adaptation and test. This is completion of an existing vocabulary, not discovery of new concepts.

Let $\mathcal H$ contain the known entries at the target receiver and $\mathcal P(\mathcal H)$ its compatible permutations. Across adaptation windows $b\in\mathcal B_A$, define
\begin{equation}
 J_A(M,\mathcal C)=\sum_{b\in\mathcal B_A}\sum_{i<j}
 a_{ij}(\mathcal C_b)w_{ij}(M),\qquad M\in\mathcal P(\mathcal H),\label{eq:partial_objective}
\end{equation}
using the original signed weights and feasible event partitions. Only the target receiver's map is variable. Given provisional groups, update
\begin{equation}
 M^{+}\in\argmax_{M\in\mathcal P(\mathcal H)}J_A(M,\mathcal C).\label{eq:partial_update}
\end{equation}
This step compares the original compatibility after each candidate reordering. It does not replace logarithmic compatibility by a different feature. With four classes and one anchor, only $3!=6$ candidates remain, so direct enumeration is sufficient; no linear-assignment claim is required.

Initialization uses the already fitted physical-only scorer to form provisional groups without relying on unknown column meanings. A sequential comparator estimates the missing map on these fixed groups and freezes it. The feedback extension re-decodes the adaptation windows with that map, keeps the previous partition whenever it has a higher objective, and repeats the map update. Thus plausible shared events provide class-correspondence evidence, and the resulting common class probabilities may change the next event association. Ties retain the preceding state; gains must exceed $10^{-10}$, and ten sweeps are allowed. Accepted changes do not decrease the adaptation objective, but need not improve reference accuracy. Learned maps are frozen before held-out windows are decoded.

The inputs to this optional module contain no target event identities or hidden map entries. Inadequate overlap, wrong provisional groups, or indistinguishable free categories can prevent recovery. The numerical probe below therefore reports withheld-entry accuracy separately from event F1 and compares feedback with the stronger sequential estimate, not only an unresolved column order.
'''

EXPERIMENT=r'''\subsection{Small probe with missing class correspondences}\label{sec:partial_probe}
This auxiliary probe leaves all preceding experiments and their figures/tables unchanged. In the controlled data, only receiver 1 has an incomplete map: one of four class correspondences is disclosed, and three are withheld. Other maps are known. The same frozen joint and physical models are reused, with no new RF scenes or classifier fitting. Prefixes of 1, 5 or 40 existing calibration scenes supply adaptation descriptors; all 800 original test scenes remain evaluation data. Calibration-descriptor reuse and intervention seeds are documented in Supplementary Section S9.

Table~\ref{tab:partial_probe} shows that one adaptation scene recovers 26 of 30 withheld entries across ten repetitions (86.7\%); five and 40 scenes recover all 30 and reproduce the original known-map F1 of 0.7798. An anchor-compatible completion without data scores 0.7026. This intervention hides one receiver's map, whereas Table~\ref{tab:semantic_alignment} tests the original all-source column-meaning assumption; the two controls have different difficulty.

For an additional measured-input check, the original public study is retained. Only in this new probe, 3 August supplies 15,254 adaptation reports and 2 September supplies 8423 test reports. At one gateway, the SF7/SF12 correspondences are hidden by a swapped native encoding; the other four SF entries and reference gateway are known. The two withheld entries are recovered, restoring test F1 from 0.1270 under unresolved order to 0.9961, equal to the original complete-map result on that capture. This is controlled code hiding on decoded metadata, not natural cross-protocol vocabulary discovery or measured classifier posteriors. Sequential fitting and feedback tie in every tested setting; the data support a bounded missing-map capability, not an extra accuracy gain from feedback.

\input{tables/partial_mapping_probe}
'''

SUPPLEMENT=r'''\section{Optional incomplete-registry probe}\label{supp:partial_probe}
\subsection{Preserved baseline and small inference wrapper}
The optional module is implemented separately in \texttt{code/partial\_mapping\_extension.py}. The original generator, entropy-weighted compatibility, scaler, coefficients, temperatures, physical gate and constrained decoder are unchanged. The observable-only interface contains receiver IDs, native probability vectors or decoded-category indicators, physical pair features and gate residuals. It contains no event reference, hidden category map, payload hash, or source coordinate. Benchmark construction and evaluation hold that information separately. Existing figures and tables remain baseline results; no baseline prediction is relabeled as a result of map learning.

The target source has $F$ unanchored native entries. Candidate maps enumerate the $F!$ unused common-class assignments after anchored rows/values are removed. Given provisional groups, each candidate is scored with the original entropy-weighted log feature and frozen model. In particular, no Hungarian update for a bilinear surrogate is substituted. A physical-only initialization avoids using arbitrary native column indices as common meanings. The sequential map estimate is a complete enumeration for the fixed initial groups. Feedback compares a fresh greedy partition with the preceding feasible partition under the current map, retaining the better one, then enumerates maps again. Strict gains exceed $10^{-10}$; ties retain the preceding state and the cap is ten sweeps. Feasibility is inherited from the original decoder. Adaptation objective ascent is not a semantic-correctness or global-optimality guarantee.

\subsection{Controlled reuse and masked entries}
For repetition $r$, scene seeds remain $R_r=10^6+10^5r$, calibration/adaptation $R_r+10{,}000+j$ ($j=0,\ldots,39$), and test $R_r+20{,}000+j$ ($j=0,\ldots,79$). The adapter first reverses the generator's declared cyclic encoding, then re-encodes receiver 1 with a permutation drawn using seed $9{,}100{,}000+r$, stable across adaptation and test. Other maps are known. The correspondence to common class 0 is disclosed; its native index depends on the hidden codebook. The remaining three entries are evaluated, for 30 withheld entries across ten repetitions. No reference instance labels are passed to the mapping module.

These calibration descriptors have already contributed labels to the historical scalar-temperature fit. Their reuse for the auxiliary mapping probe is therefore not claimed as a new independent calibration/adaptation collection. All original fitted parameters are frozen, mapping choices use only the re-encoded descriptors and provisional groups, and test references never select an update. The separate public probe has a later adaptation capture. Both analyses are exploratory extensions on previously inspected data.

Mapping accuracy excludes the anchor and all known receivers. Controlled F1 first averages the 80 scene F1 values within each repetition, then averages the ten repetition means. The known-map adapter reproduces all 800 original scene F1 values. Five- and 40-scene adaptation recover all three withheld entries in every repetition; one-scene adaptation recovers 26/30. Sequential and feedback test outcomes tie. Raw predictions, maps, traces and replication summaries are supplied, including the five-scene condition omitted from the compact main table.

\subsection{Public code hiding and independent later-capture evaluation}
The original six-file LoED filtering, clock corrections, empirical gate, timing scorer and three-feature metadata scorer are retained. The two active original receiver IDs are 0 and 3; local IDs 0 and 1 only simplify the wrapper. The 3 August 2020 capture supplies 15,254 adaptation reports, while 2 September supplies 8423 test reports. Their reference identities do not overlap. At local receiver 1, native SF7/SF12 column meanings are exchanged and withheld, while SF8--SF11 correspondences are fixed. The reference source retains its known SF7--SF12 vocabulary. There are only two candidate maps. Decoded SFs are categorical metadata, not noisy protocol-recognition posteriors; measured time and frequency values are not perturbed.

Initialization uses the original timing-only model. Map candidates and subsequent association use the original metadata model, whose third feature is SF mismatch. Physical features contain only corrected-time residual and reported-frequency mismatch. Payload references are isolated from mapping and decoding. Both withheld entries are recovered. On the later test capture, the learned map yields TP 511, FP 1 and FN 3, exactly matching the complete-map reference. Under unresolved order the counts are TP 35, FP 2 and FN 479. Both methods return feasible partitions. This two-entry correction establishes a limited measured-input mechanism; it cannot establish unknown protocol semantics, recovery of a completely hidden six-class map, or a general benefit over sequential alignment.

Run \texttt{python code/partial\_mapping\_extension.py} and then the accompanying verification script. Results are under \texttt{results/partial\_mapping\_extension/}. Separate mapping and event metrics, objective traces and observation-level predictions make it possible to distinguish missing-registry recovery from a change in the original classifier or decoder.
'''

def main():
    replace('main.tex','The method uses known mappings rather than learning an ontology.',
        'The baseline uses declared mappings; a small extension learns missing entries through event-association feedback.')
    replace('sections/01_introduction.tex',
        'We do not discover unknown vocabulary mappings, learn hierarchical concepts, or classify waveforms.',
        'A limited extension estimates missing entries of a fixed class map; unknown field meanings, hierarchical concepts and waveform classification remain outside scope.')
    replace('sections/01_introduction.tex',
        'and a separately scoped public-recording check with complete inclusion and prediction records.',
        'a separately scoped public-recording check with complete inclusion and prediction records, and a small incomplete-registry probe.')
    replace('sections/01_introduction.tex',
        'It therefore tests measured metadata and reference-packet correspondence, while the class-semantic mechanism remains a controlled validation.',
        'It therefore tests measured metadata and reference-packet correspondence. The added probe hides two decoded SF correspondences; soft class probabilities remain a controlled validation.')
    replace('sections/03_methodology.tex',
        'Automatic discovery of unknown field meanings or class mappings is outside this implementation.',
        r'The baseline assumes complete class registries. Section~\ref{sec:partial_registry} estimates selected missing class entries; unknown field meanings remain outside this implementation.')
    replace('sections/03_methodology.tex',r'\FloatBarrier',r'\input{sections/03a_partial_mapping}'+'\n'+r'\FloatBarrier')
    (ROOT/'sections/03a_partial_mapping.tex').write_text(METHOD,encoding='utf-8')
    replace('sections/04_experiments.tex',
        'Vocabulary-mapping accuracy is not reported because mappings are supplied rather than learned.',
        r'The main experiments assume supplied maps; the auxiliary probe in Section~\ref{sec:partial_probe} separately reports withheld-entry accuracy.')
    replace('sections/04_experiments.tex',r'\FloatBarrier',r'\input{sections/04c_partial_mapping}'+'\n'+r'\FloatBarrier')
    (ROOT/'sections/04c_partial_mapping.tex').write_text(EXPERIMENT,encoding='utf-8')
    replace('sections/04b_discussion.tex',
        '\\subsection{Role and cost of consistency in instance alignment}',
        r'The optional incomplete-registry probe relaxes only selected class correspondences at one receiver. Plausible event groups supply evidence for completing the map, and the mapped probabilities are then available for event re-association. Other meanings remain declared. Recovery in this bounded setting does not imply ontology discovery. Sequential estimation and feedback tie in the executed conditions, so the added evidence supports missing-map completion rather than an additional feedback accuracy gain.'+'\n\n'+r'\subsection{Role and cost of consistency in instance alignment}')
    replace('sections/05_conclusions.tex',
        'while retaining the assumption that semantic mappings are known.',
        'with known semantic mappings in the main pipeline and a limited extension for selected missing class entries.')
    replace('sections/05_conclusions.tex',
        'Future progress requires source-specific semantic annotations,',
        'The optional extension recovers withheld entries within a fixed vocabulary, with sequential and feedback methods tied in the tested settings. Future progress requires source-specific semantic annotations,')
    replace('sections/05_conclusions.tex',
        'declared-mapping audit outputs, and verification reports.',
        'declared-mapping audit outputs, optional missing-map probe predictions and traces, and verification reports.')
    replace('supplement.tex',r'\end{document}',SUPPLEMENT+'\n'+r'\end{document}')
    controlled=pd.read_csv(ROOT/'results/partial_mapping_extension/controlled_summary.csv')
    public=pd.read_csv(ROOT/'results/partial_mapping_extension/public_summary.csv')
    names={'completion':'Unresolved completion','sequential':'Sequential map fit','feedback':'Map--event feedback','known':'Complete-map reference'}
    lines=[r'\begin{table}[!t]\begin{adjustwidth}{-\extralength}{0cm}',
        r'\caption{Optional missing-map probe using the original scores and decoder. Controlled results average ten repetitions; public results describe one later test capture. Mapping accuracy counts withheld entries only.}\label{tab:partial_probe}',
        r'\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.10}',
        r'\begin{tabularx}{\linewidth}{@{}lYrr@{}}\toprule',
        r'Probe / adaptation & Method & Map (\%) & Event F1\\\midrule']
    for budget,methods in [(1,['completion','sequential','feedback']),(40,['sequential','feedback','known'])]:
        for method in methods:
            row=controlled[(controlled.budget==budget)&(controlled.method==method)].iloc[0]
            lines.append(f'Controlled / {budget} scene'+('s' if budget>1 else '')+f' & {names[method]} & {100*row.mapping_accuracy:.1f} & {row.f1:.4f}'+r'\\')
    lines.append(r'\midrule')
    for method in ['completion','sequential','feedback','known']:
        row=public[public.method==method].iloc[0]
        lines.append(f'Public / 1 capture & {names[method]} & {100*row.mapping_accuracy:.1f} & {row.f1:.4f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabularx}',
        r'\par\vspace{3pt}\footnotesize Controlled: three hidden entries at one receiver; public: only SF7/SF12 at one gateway. Five-scene controlled adaptation also reaches 100\% and F1 0.7798. The public test contains 8423 reports, rather than the 23,677 reports pooled in the original public study. No improvement over sequential fitting is observed.',
        r'\end{adjustwidth}\end{table}'])
    (ROOT/'tables/partial_mapping_probe.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Applied bounded additions; existing figure assets and table files were not edited.')

if __name__=='__main__':main()
