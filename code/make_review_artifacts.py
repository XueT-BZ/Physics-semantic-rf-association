"""Publication tables and figures derived only from saved review-audit results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_common import ROOT,OUT

TABLES=ROOT/'tables';FIG=ROOT/'figures'

def table(name,cols,header,rows):
    lines=[r'\begin{tabularx}{\linewidth}{@{}'+cols+r'@{}}\toprule',header+r' \\\midrule']
    lines.extend(' & '.join(row)+r' \\' for row in rows)
    lines.append(r'\bottomrule\end{tabularx}')
    (TABLES/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def f(x,d=4):return f'{float(x):.{d}f}'
def ci(row,name='f1'):return f(row[name])+r' ['+f(row[name+'_lo'])+', '+f(row[name+'_hi'])+']'
def pct(x):return f'{100*max(0.,float(x)):.1f}'+r'\%'

def main():
    factorial=pd.read_csv(OUT/'factorial/summary.csv')
    scorer_names={'physics':'Physical linear','joint':'Joint linear','unweighted':'Unweighted joint','boosted':'Boosted joint'}
    decoder_names={'components':'Components','assignment':'Assignment + union','constrained':'Constrained'}
    rows=[]
    for scorer in scorer_names:
        for decoder in decoder_names:
            r=factorial[(factorial.scorer==scorer)&(factorial.decoder==decoder)].iloc[0]
            rows.append([scorer_names[scorer],decoder_names[decoder],ci(r),f(r.ari),f(r.b3f1),f(r.exact),pct(r.receiver_violation),pct(r.gate_violation)])
    table('factorial.tex','llYrrrrr',r'Scorer & Decoder & Event-pair F1 [95\% CI] & ARI & $B^3$ F1 & Exact & Rx conflict & Gate violation',rows)
    complete=pd.read_csv(OUT/'mapping/completion_summary.csv')
    coordinate=pd.read_csv(OUT/'mapping_coordinate/completion_summary.csv')
    wrong=pd.read_csv(OUT/'mapping/wrong_map_summary.csv')
    rows=[]
    for budget in [1,5,40]:
        for sources in [1,2,3]:
            a=complete[(complete.sources==sources)&(complete.anchors==0)&(complete.budget==budget)&(complete.method=='feedback')].iloc[0]
            s=complete[(complete.sources==sources)&(complete.anchors==0)&(complete.budget==budget)&(complete.method=='sequential')].iloc[0]
            c=coordinate[(coordinate.sources==sources)&(coordinate.anchors==0)&(coordinate.budget==budget)&(coordinate.method=='feedback')].iloc[0]
            rows.append([str(budget),str(sources),f'{int(a.withheld_correct)}/{int(a.withheld_total)}',f(s.f1),ci(a),f(c.f1),r'0 / 0'])
    table('review_mapping.tex','rrrYrYr',r'Adaptation scenes & Unknown sources & Correct hidden entries & Sequential F1 & Feedback F1 [95\% CI] & Coordinate F1 & Rx / gate',rows)
    # All anchor settings remain available in a supplementary table.
    rows=[]
    for sources in [1,2,3]:
        for anchors in [0,1,2]:
            for budget in [1,5,40]:
                a=complete[(complete.sources==sources)&(complete.anchors==anchors)&(complete.budget==budget)&(complete.method=='feedback')].iloc[0]
                rows.append([str(sources),str(anchors),str(budget),f'{int(a.withheld_correct)}/{int(a.withheld_total)}',ci(a),f(a.exact)])
    table('review_mapping_full.tex','rrrrYr',r'Unknown sources & Anchors / source & Adaptation scenes & Correct / hidden & Event-pair F1 [95\% CI] & Exact',rows)
    risks=pd.read_csv(OUT/'risks/summary.csv')
    risk_names={'nominal':'Nominal subset','positive_corr_independent_declaration':r'Positive corr.; diagonal declaration',
        'positive_corr_matched':r'Positive corr.; matched covariance','negative_corr_independent_declaration':r'Negative corr.; diagonal declaration',
        'negative_corr_matched':r'Negative corr.; matched covariance','nlos_excess_delay':'Extra positive delay',
        'independent_false_alarms_matched_count':'Independent false alarms; same count',
        'correlated_false_alarms':'Correlated false alarms','fragmented_detections':'Fragmented reports'}
    rows=[]
    joint=risks[(risks.scorer=='joint')&(risks.decoder=='constrained')].set_index('condition')
    for name,title in risk_names.items():
        a=joint.loc[name]
        rows.append([title,ci(a),f(a.true_pair_gate_recall),f(a.clutter_attachment),f(a.exclusivity_recall_ceiling),r'0 / 0'])
    table('review_risks.tex','YYrrrr',r'Intervention & Event-pair F1 [95\% CI] & Gate recall & Clutter attachment & Recall ceiling & Rx / gate',rows)
    exact=pd.read_csv(OUT/'exact/summary.csv');rows=[]
    for _,a in exact.iterrows():
        rows.append([str(int(a.observations)),f'{int(a.proven_optimal)}/{int(a.instances)}',pct(a.greedy_gap_mean),pct(a.restart_gap_mean),
                     f(a.greedy_f1),f(a.restart_f1),f(a.milp_f1_proven),f(a.greedy_ms,2)+' / '+f(a.restart_ms,2)+' / '+f(a.milp_ms,2)])
    table('review_exact.tex','rrrrrrrY',r'$N$ & Proven optima & Greedy gap & Restart gap & Greedy F1 & Restart F1 & MILP F1 & Median ms: greedy / restart / MILP',rows)
    public=pd.read_csv(OUT/'public_sensitivity/summary.csv')
    captures=pd.read_csv(OUT/'public_sensitivity/per_capture.csv')
    statuses=pd.read_csv(OUT/'public_sensitivity/setting_status.csv').set_index('setting')
    public_names={'original_split':'Original 3/1 split','first_two_train':'2019 training blocks only','latest_train_only':'May training only',
        'second_train_only':'April training only','early_calibration':'March train / April calibrate','middle_calibration':'2019 train / May calibrate',
        'uncorrected_clocks':'Clock correction disabled','gate_quantile_095':'Calibration quantile 0.95','gate_quantile_0999':'Calibration quantile 0.999'}
    rows=[]
    for name,title in public_names.items():
        s=statuses.loc[name]
        if s.status=='evaluated':
            a=public[(public.setting==name)&(public.scorer=='metadata')].iloc[0]
            ds=captures[(captures.setting==name)&(captures.scorer=='metadata')].sort_values('day')
            rows.append([title,str(int(s.training_positive_pairs)),f(1000*a.gate_s,3),f(ds.iloc[0].f1),f(ds.iloc[1].f1),f(a.f1)])
        else:rows.append([title,'0' if 'training' in s.status else '--','--','NE','NE','NE'])
    table('review_public_sensitivity.tex','Yrrrrr',r'Preprocessing setting & Train positives & $G$ (ms) & Aug. F1 & Sep. F1 & Pooled F1',rows)
    pooled=pd.read_csv(OUT/'factorial/pooled_pair_counts.csv').groupby(['scorer','decoder'],as_index=False)[['tp','fp','fn','true_pairs']].sum()
    rows=[]
    for _,a in pooled.iterrows():
        m=factorial[(factorial.scorer==a.scorer)&(factorial.decoder==a.decoder)].iloc[0]
        micro=2*a.tp/(2*a.tp+a.fp+a.fn)
        rows.append([scorer_names[a.scorer],decoder_names[a.decoder],str(int(a.tp)),str(int(a.fp)),str(int(a.fn)),f(m.f1),f(micro)])
    table('review_pair_counts.tex','llrrrrr',r'Scorer & Decoder & TP & FP & FN & Scene-macro F1 & Pair-micro F1',rows)
    # Existing public results receive explicit counts and feasibility in the supplement.
    old_public=pd.read_csv(ROOT/'results/loed/per_day.csv');rows=[]
    for _,a in old_public.iterrows():
        rows.append([a.day,a.scorer,decoder_names[a.decoder],str(int(a.tp)),str(int(a.fp)),str(int(a.fn)),f(a.f1),str(int(a.receiver_conflicts)),str(int(a.gate_violations))])
    table('review_public_counts.tex','lllrrrrrr',r'Capture & Scorer & Decoder & TP & FP & FN & Reference-packet F1 & Rx & Gate',rows)

    risk_full=pd.read_csv(OUT/'risks/summary.csv')
    full=[r'\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{@{}p{3.8cm}llrrrrrr@{}}',
          r'\caption{All assumption-test accuracy and feasibility metrics. Rx and Gate are violation percentages.}\label{tab:all_risk_metrics}\\',
          r'\toprule Intervention & Scorer & Decoder & F1 & ARI & $B^3$ F1 & Exact & Rx & Gate\\\midrule\endfirsthead',
          r'\toprule Intervention & Scorer & Decoder & F1 & ARI & $B^3$ F1 & Exact & Rx & Gate\\\midrule\endhead',
          r'\bottomrule\endfoot']
    for _,a in risk_full.iterrows():
        short={'components':'CC','assignment':'Assign.','constrained':'Constr.'}[a.decoder]
        full.append(' & '.join([risk_names[a.condition],{'joint':'Joint','physics':'Phys.'}[a.scorer],short,
                 f(a.f1),f(a.ari),f(a.b3f1),f(a.exact),pct(a.receiver_violation),pct(a.gate_violation)])+r' \\')
    full.extend([r'\end{longtable}',r'\normalsize'])
    (TABLES/'review_risks_full.tex').write_text('\n'.join(full)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.22,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5))
    ax=axes[0]
    ax.errorbar(wrong.wrong_fraction,wrong.f1,yerr=np.vstack([wrong.f1-wrong.f1_lo,wrong.f1_hi-wrong.f1]),fmt='o-',capsize=3,label='Independent map errors')
    physical_reference=float(factorial[(factorial.scorer=='physics')&(factorial.decoder=='constrained')].iloc[0].f1)
    ax.axhline(physical_reference,ls='--',color='#777',lw=1,label='Refitted physical-only')
    ax.scatter([1.],[wrong.iloc[0].f1],marker='*',s=100,color='#a33737',zorder=5,label='Shared renaming control')
    ax.set(xlabel='Wrong entries / all 16 entries',ylabel='Event-pair F1',ylim=(.43,.84),xticks=[0,.25,.5,.75,1])
    ax.legend(fontsize=8,loc='lower left');ax.set_title('(a) Wrong mapping entries')
    for budget,color,style,mark in [(1,'#0072b2','-','o'),(5,'#009e73','-','s'),(40,'#cc79a7','--','x')]:
        data=complete[(complete.anchors==0)&(complete.budget==budget)&(complete.method=='feedback')].sort_values('sources')
        x=data.sources/4
        axes[1].plot(x,data.mapping_accuracy,style+mark,color=color,label=f'{budget} adaptation scenes')
        axes[2].errorbar(x,data.f1,yerr=np.vstack([data.f1-data.f1_lo,data.f1_hi-data.f1]),fmt=style+mark,color=color,capsize=3,label=f'{budget} scenes')
    axes[1].set(xlabel='Missing entries / all 16 entries',ylabel='Hidden-entry accuracy',ylim=(.45,1.04),xticks=[.25,.5,.75])
    axes[1].set_title('(b) Missing maps; no target anchors');axes[1].legend(fontsize=8,loc='lower left')
    axes[2].axhline(wrong.iloc[0].f1,ls=':',color='#777',lw=1)
    axes[2].set(xlabel='Missing entries / all 16 entries',ylabel='Event-pair F1',ylim=(.56,.84),xticks=[.25,.5,.75])
    axes[2].set_title('(c) Joint map completion')
    fig.tight_layout();fig.savefig(FIG/'review_mapping.pdf',bbox_inches='tight');fig.savefig(FIG/'review_mapping.png',dpi=180,bbox_inches='tight');plt.close(fig)
    labels=['Nominal','Positive corr. / diagonal','Positive corr. / matched','Negative corr. / diagonal','Negative corr. / matched','Extra delay','Independent alarms / same count','Correlated false alarms','Fragments']
    order=list(risk_names);y=np.arange(len(order));fig,axes=plt.subplots(1,2,figsize=(10,4.1),sharey=True)
    values=joint.loc[order]
    axes[0].barh(y,values.f1,color='#4389b7',height=.58)
    axes[0].errorbar(values.f1,y,xerr=np.vstack([values.f1-values.f1_lo,values.f1_hi-values.f1]),fmt='none',ecolor='#222',capsize=2)
    axes[0].set(xlabel='Event-pair F1',xlim=(.45,.85),yticks=y,yticklabels=labels);axes[0].invert_yaxis()
    axes[0].set_title('(a) Joint constrained association')
    axes[1].plot(values.true_pair_gate_recall,y,'o',label='True-pair gate recall',color='#0072b2')
    axes[1].plot(values.exclusivity_recall_ceiling,y,'x',label='Exclusivity recall ceiling',color='#d55e00')
    axes[1].set(xlabel='Recall / upper bound',xlim=(.58,1.03));axes[1].set_title('(b) Admission versus representability')
    axes[1].legend(fontsize=8,loc='lower left');fig.tight_layout()
    fig.savefig(FIG/'review_risks.pdf',bbox_inches='tight');fig.savefig(FIG/'review_risks.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Machine-readable facts for manuscript and rebuttal verification.
    facts=dict(wrong_map_f1={str(a.wrong_fraction):float(a.f1) for _,a in wrong.iterrows()},
               joint_pair_counts=pooled[(pooled.scorer=='joint')&(pooled.decoder=='constrained')].iloc[0].to_dict(),
               exact_instances=int(exact.instances.sum()),exact_proven=int(exact.proven_optimal.sum()),
               public_evaluated_settings=int((statuses.status=='evaluated').sum()),
               public_non_estimable_settings=int((statuses.status!='evaluated').sum()))
    (OUT/'manuscript_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    print('Generated six main/review tables, supplementary metric tables, and two vector figures.')

if __name__=='__main__':main()
