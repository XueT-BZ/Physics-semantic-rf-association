"""Render measured-data tables from saved full-precision results."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def emit(name,columns,header,rows):
    lines=['\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}'+columns+'@{}}\\toprule',
           header+' '+chr(92)*2+r'\midrule']
    lines.extend(row+' '+chr(92)*2 for row in rows)
    lines.append(r'\bottomrule\end{tabular*}')
    (ROOT/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


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
    print('Three measured-data tables rebuilt from results/loed/.')


if __name__=='__main__':main()
