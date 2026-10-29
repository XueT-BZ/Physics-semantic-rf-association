"""Render public-data and original missing-map tables from saved numeric results."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def emit(name,columns,header,rows):
    lines=['\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}'+columns+'@{}}\\toprule',
           header+' '+chr(92)*2+r'\midrule']
    lines.extend(row+' '+chr(92)*2 for row in rows)
    lines.append(r'\bottomrule\end{tabular*}')
    (ROOT/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def partial_mapping_table():
    """Export the retained S9 probe without editing manuscript paragraphs."""
    controlled=pd.read_csv(ROOT/'results/partial_mapping_extension/controlled_summary.csv')
    public=pd.read_csv(ROOT/'results/partial_mapping_extension/public_summary.csv')
    names={'completion':'Unresolved completion','sequential':'Sequential map fit',
           'feedback':'Map--event feedback','known':'Complete-map reference'}
    lines=[r'\begin{table}[!t]\begin{adjustwidth}{-\extralength}{0cm}',
        r'\caption{Optional missing-map test with the original scores and decoder. Controlled results average ten repetitions. Public results use one later test capture. Mapping accuracy counts only hidden entries. Controlled rows use event-pair F1. Public rows use reference-packet F1.}\label{tab:partial_probe}',
        r'\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.10}',
        r'\begin{tabularx}{\linewidth}{@{}lYrr@{}}\toprule',
        r'Probe / adaptation & Method & Map (\%) & Pair F1\\\midrule']
    for budget,methods in [(1,['completion','sequential','feedback']),
                           (40,['sequential','feedback','known'])]:
        for method in methods:
            row=controlled[(controlled.budget==budget)&(controlled.method==method)].iloc[0]
            label=f'Controlled / {budget} scene'+('s' if budget>1 else '')
            lines.append(f'{label} & {names[method]} & {100*row.mapping_accuracy:.1f} & {row.f1:.4f}'+r'\\')
    lines.append(r'\midrule')
    for method in ['completion','sequential','feedback','known']:
        row=public[public.method==method].iloc[0]
        lines.append(f'Public / 1 capture & {names[method]} & {100*row.mapping_accuracy:.1f} & {row.f1:.4f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabularx}',
        r'\par\vspace{3pt}\footnotesize The controlled test hides three entries at one receiver. The public test hides only SF7/SF12 at one gateway. Five-scene controlled adaptation also reaches 100\% and F1 0.7798. The public test has 8423 reports. The original public study pools 23,677 reports. Feedback does not improve on sequential fitting.',
        r'\end{adjustwidth}\end{table}'])
    (ROOT/'tables/partial_mapping_probe.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    inc=pd.read_csv(ROOT/'results/loed/inclusion.csv')
    rows=[]
    for split,label in [('train','Training'),('calibration','Calibration'),('test','Test')]:
        g=inc[inc.split==split]
        rows.append(f'{label} & {len(g)} & {g.raw_rows.sum():,} & {g.eligible_reports.sum():,} & {g.visible_reference_packets.sum():,}')
    emit('loed_inclusion','lrrrr','Split & Files & Raw reports & Eligible reports & Visible reference packets',rows)
    day=pd.read_csv(ROOT/'results/loed/per_day.csv')
    pooled=pd.read_csv(ROOT/'results/loed/summary.csv')
    rows=[]
    for kind,label in [('timing','Timing'),('metadata','Metadata')]:
        for decoder,dlabel in [('components','Components'),('assignment','Assignment'),('constrained','Constrained')]:
            g=day[(day.scorer==kind)&(day.decoder==decoder)].sort_values('day')
            p=pooled[(pooled.scorer==kind)&(pooled.decoder==decoder)].iloc[0]
            rows.append(f'{label} & {dlabel} & {g.iloc[0].f1:.4f} & {g.iloc[1].f1:.4f} & {p.f1:.4f} & {int(p.receiver_conflicts)}')
    emit('loed_results','llrrrr','Evidence & Decoder & 3 Aug. F1 & 2 Sep. F1 & Pooled F1 & Rx',rows)
    rows=[f'{a.day} & {a.raw_rows:,} & {a.valid_unconfirmed_uplink_rows:,} & {a.exact_duplicate_rows_removed:,} & {a.ambiguous_rows_removed:,} & {a.eligible_reports:,}' for a in inc.itertuples()]
    emit('loed_files','lrrrrr','Capture file date & Raw & Valid uplinks & Exact duplicates & Ambiguous & Eligible',rows)
    partial_mapping_table()
    print('Public-data and original missing-map tables rebuilt from saved results.')


if __name__=='__main__':main()
