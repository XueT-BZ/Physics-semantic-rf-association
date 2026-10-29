#!/usr/bin/env python3
"""Render publication figures and tables from executed outputs, never digitized images.

Run after experiment.py and aggregate.py. Scientific data are kept in results/;
curve samples used by the figures are exported to results/figure_data/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
R, F, T = ROOT/'results', ROOT/'figures', ROOT/'tables'
D = R/'figure_data'
for path in (F, T, D):
    path.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.2,
    'axes.labelsize':9.2, 'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
    'legend.fontsize':8.1, 'figure.dpi':150, 'savefig.dpi':240,
    'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
    'axes.spines.right':False})
P = pd.read_csv(R/'pair_summary.csv').set_index('method')
C = pd.read_csv(R/'cluster_summary.csv').set_index('method')
S = pd.read_csv(R/'stress_summary.csv')
G = pd.read_csv(R/'gate_summary.csv')
labels = {
 'time':'Timing only', 'frequency':'Frequency only', 'semantic':'Semantics only',
 'physics':'Physical, ungated', 'joint':'Joint linear, ungated',
 'boosted':'Boosted joint, ungated', 'raw_units':'Raw-frequency feature',
 'unweighted':'Unweighted semantics, ungated', 'joint_gated':'Joint linear, gated',
 'joint_calibrated':'Joint linear, gated + $T$',
 'physics_calibrated':'Physical, gated + $T$',
 'boosted_calibrated':'Boosted joint, gated + $T$'}
cl = {
 'physics_constrained':'Physical, constrained', 'boosted_constrained':'Boosted, constrained',
 'joint_components':'Joint, connected components', 'pairwise_assignment':'Pairwise assignment + union',
 'joint_constrained':'Joint, constrained', 'no_receiver_exclusion':'Without receiver exclusivity',
 'no_complete_gate':'Without all-pair gate check', 'unweighted_semantics':'Unweighted, constrained'}

def save(fig, name):
    fig.savefig(F/f'{name}.pdf', bbox_inches='tight', pad_inches=.025)
    fig.savefig(F/f'{name}.png', bbox_inches='tight', pad_inches=.025)
    plt.close(fig)

def axpair():
    fig, ax = plt.subplots(figsize=(3.58, 2.70))
    fig.subplots_adjust(left=.17, right=.985, bottom=.19, top=.97)
    ax.grid(alpha=.23, linewidth=.6)
    return fig, ax

def zoompair(figsize=(3.65, 3.36), heights=(2.9, 1.0)):
    """Show exact trajectories above and a separately scaled, data-derived difference below."""
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': list(heights)})
    fig.subplots_adjust(left=.19, right=.975, top=.96, bottom=.155, hspace=.075)
    for a in axs:
        a.grid(alpha=.25, linewidth=.6)
        a.set_axisbelow(True)
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelsize=7.3)
    return fig, axs

def table(name, header, rows, align):
    (T/f'{name}.tex').write_text(
      '\\begin{tabularx}{\\linewidth}{@{}'+align+'@{}}\n\\toprule\n'+header+
      ' \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabularx}\n', encoding='utf-8')

def ci(row, key, digits=4):
    return f"{row[key]:.{digits}f} [{row[key+'_lo']:.{digits}f}, {row[key+'_hi']:.{digits}f}]"

rows = []
for m in labels:
    r = P.loc[m]
    rows.append(f"{labels[m]} & {r.auroc:.4f} & {ci(r, 'ap')} & {r.brier:.4f} & {r.nll:.4f} \\\\")
table('pair', 'Method & AUROC & AP [95\\% CI] & Brier & NLL', rows, 'Yrrrr')
rows = []
for m in cl:
    r = C.loc[m]
    rows.append(f"{cl[m]} & {ci(r, 'f1')} & {r.ari:.4f} & {r.exact:.4f} & {100*r.receiver_violation:.2f} & {100*r.gate_violation:.3f} \\\\")
table('cluster', 'Scoring and decoding & F1 [95\\% CI] & ARI & Exact & Rx (\\%) & Gate (\\%)', rows, 'Yrrrrr')
scenarios = {
 'nominal':'Nominal stress reference', 'dense':'28 events',
 'missed_detections':'Detection probability 0.55', 'clutter':'Clutter mean 2.5 / receiver',
 'semantic_error':'Semantic error probability 0.45', 'semantic_missing':'Missing semantics probability 0.65',
 'clock_misspecified':'Clock s.d. $6\\,\\mu$s; declared $0.8\\,\\mu$s',
 'frequency_misspecified':'Frequency bias s.d. 1.5 kHz; declared 0.25 kHz',
 'multipath':'Additional positive timing delays', 'heavy_tails':'Variance-matched $t_3$ timing noise',
 'unseen_receiver':'Fifth receiver, not used in training',
 'overconfident_wrong':'One confidently wrong receiver'}
rows = []
for v, label in scenarios.items():
    if v=='nominal':rows.append(r'\multicolumn{4}{l}{\textit{Reference condition on the stress seeds}}\\')
    data = S[(S.group=='scenario') & (S.value==v)].set_index('method')
    p, j = data.loc['physics'], data.loc['joint']
    rows.append(f"{label} & {p.f1:.4f} & {ci(j, 'f1')} & {j.gate_recall:.4f} \\\\")
    if v=='nominal':
        rows.extend([r'\midrule',r'\multicolumn{4}{l}{\textit{Interventions on the same seed schedule}}\\'])
table('stress', 'Test condition & Physical event F1 & Joint event F1 [95\\% CI] & Gate recall', rows, 'Yrrr')
rows = [f"{r.k:g} & {r.gate_recall:.4f} & {r.retained_fraction:.4f} & {ci(r,'f1')} \\\\" for _, r in G.iterrows()]
table('gate', '$k$ & True-pair recall & Retained pair fraction & Event-pair F1 [95\\% CI]', rows, 'Yrrr')

# Curves: one explicitly labeled held-out replication, not ten-run averages.
# Match the height of the adjacent two-level reliability diagnostic.
fig, ax = plt.subplots(figsize=(3.62,3.36))
fig.subplots_adjust(left=.17,right=.98,bottom=.16,top=.96)
ax.grid(alpha=.23,linewidth=.6)
prrows = []
for m, label, style in [('physics_calibrated','Physical','-'),
                        ('joint_calibrated','Joint linear','--'),
                        ('boosted_calibrated','Boosted joint','-.')]:
    data = np.load(R/f'prediction_{m}_r00.npz')
    precision, recall, thresholds = precision_recall_curve(data['y'], data['prob'])
    ax.plot(recall, precision, linestyle=style, label=label, linewidth=1.5)
    prrows.extend({'method':m,'recall':float(r),'precision':float(p)} for p,r in zip(precision,recall))
ax.set(xlabel='Recall', ylabel='Precision', xlim=(0,1), ylim=(0,1.025))
ax.legend(loc='lower left', frameon=False)
save(fig, 'results_pr')
pd.DataFrame(prrows).to_csv(D/'precision_recall.csv', index=False)

# Figure 8b: the original temperature-scaling trajectories nearly coincide.
# Keep their original values, and show the *true bin-wise difference* in a magnified axis.
fig, (ax, ad) = zoompair((3.62,3.36), (2.95,1.15))
relrows = []
bin_vals = {}
for m, label, marker, style in [('joint_gated','Before scaling','o','-'),
                                ('joint_calibrated','After scaling','s','--')]:
    data = np.load(R/f'prediction_{m}_r00.npz')
    y, p = data['y'][data['gate']], data['prob'][data['gate']]
    bins = np.minimum((10*p).astype(int),9)
    xs, ys, bb = [], [], []
    for b in range(10):
        mask = bins == b
        if mask.any():
            xx, yy = float(p[mask].mean()), float(y[mask].mean())
            xs.append(xx); ys.append(yy); bb.append(b)
            relrows.append({'method':m,'bin':b,'n':int(mask.sum()),
                            'mean_probability':xx,'positive_fraction':yy})
    # Lines are at the exact measured bin means. Marker-only offsets (0.006)
    # make nearly coincident symbols legible; source values are unchanged.
    ax.plot(xs,ys,linestyle=style,linewidth=1.1,label=label)
    shift=-.006 if m=='joint_gated' else .006
    ax.scatter(np.array(xs)+shift,ys,marker=marker,s=17,zorder=4,
               facecolor='white' if m=='joint_gated' else None)
    bin_vals[m]=dict(zip(bb,ys))
ax.plot([0,1],[0,1],linestyle=':',linewidth=.9,color='.45',label='Ideal')
ax.set(ylabel='Positive fraction',xlim=(0,1),ylim=(0,1.03))
ax.legend(loc='upper left',frameon=False,fontsize=7.3)
bb=sorted(set(bin_vals['joint_gated'])&set(bin_vals['joint_calibrated']))
xx=np.array([(b+.5)/10 for b in bb])
diff=np.array([bin_vals['joint_calibrated'][b]-bin_vals['joint_gated'][b] for b in bb])
ad.axhline(0,color='.4',lw=.75)
ad.plot(xx,diff,marker='D',markersize=3.1,linewidth=1.1)
lim=max(.012,float(np.max(np.abs(diff)))*1.35)
ad.set(xlabel='Mean predicted probability',ylabel=r'$\Delta$ obs. frac.',ylim=(-lim,lim),xlim=(0,1))
ad.yaxis.set_major_locator(plt.MaxNLocator(3))
ad.tick_params(axis='y',labelsize=7.0)
save(fig, 'results_reliability')
pd.DataFrame(relrows).to_csv(D/'reliability.csv',index=False)
pd.DataFrame({'bin':bb,'bin_center':xx,'delta_after_minus_before':diff}).to_csv(D/'reliability_difference.csv',index=False)


fig, ax = plt.subplots(figsize=(7.18,3.12))
fig.subplots_adjust(left=.355,right=.985,bottom=.17,top=.97)
order = ['physics_constrained','joint_components','no_receiver_exclusion','pairwise_assignment',
         'no_complete_gate','joint_constrained','unweighted_semantics','boosted_constrained']
r = C.loc[order]
y = np.arange(len(r))
ax.errorbar(r.f1, y, xerr=[r.f1-r.f1_lo,r.f1_hi-r.f1], fmt='o', capsize=3,
            elinewidth=1.3, markersize=4.5)
ax.set_yticks(y,[cl[m] for m in order]); ax.invert_yaxis()
ax.set(xlabel='Mean event-pair F1 with 95% confidence interval', xlim=(.63,.86))
ax.grid(axis='x', alpha=.23, linewidth=.6)
save(fig, 'results_partition')
r.to_csv(D/'partition_forest.csv')

# Figure 10a: true curves (including the crossing) and an independently
# scaled signed-performance difference reveal where joint evidence reverses.
fig,(ax,ad)=zoompair((3.62,3.36),(2.9,1.05))
sem_series={}
for m,label,marker,style in [('physics','Physical','o','-'),('joint','Joint linear','s','--')]:
    rr=S[(S.group=='semantic')&(S.method==m)].copy()
    rr['x']=rr.value.astype(float);rr=rr.sort_values('x')
    x=rr.x.to_numpy(); y=rr.f1.to_numpy()
    erlo=y-rr.f1_lo.to_numpy(); erhi=rr.f1_hi.to_numpy()-y
    ax.plot(x,y,linestyle=style,linewidth=1.35,label=label)
    # Only marker positions are nudged (not numerical x, line locations or y).
    shift=-.007 if m=='physics' else .007
    ax.errorbar(x+shift,y,yerr=[erlo,erhi],fmt=marker,linestyle='none',
                markersize=3.7,capsize=2.0,elinewidth=.8)
    sem_series[m]=(x,y)
ax.set(ylabel='Event-pair F1',ylim=(.45,.91),xlim=(-.025,.77))
ax.legend(loc='upper right',frameon=False,fontsize=7.4)
x,p=sem_series['physics'];xx,j=sem_series['joint']
assert np.allclose(x,xx)
ad.axhline(0,color='.42',linewidth=.8)
ad.plot(x,j-p,'D-',linewidth=1.1,markersize=3.2)
ad.set(ylabel=r'$\Delta$ F1 (joint-phys.)',xlabel='Incorrect-label probability',xlim=(-.025,.77))
ad.set_ylim(min(-.12,float(np.min(j-p))-.02),max(.2,float(np.max(j-p))+.025))
ad.yaxis.set_major_locator(plt.MaxNLocator(3))
save(fig,'results_semantic')
S[S.group=='semantic'].to_csv(D/'semantic_shift.csv',index=False)
pd.DataFrame({'error_probability':x,'joint_minus_physical_f1':j-p}).to_csv(D/'semantic_difference.csv',index=False)


# Figure 10b/11a: all values and model uncertainties stay unchanged.
# Report matched-minus-underreported in a separate axis; symbol-only
# horizontal nudges of +/-0.12 microseconds resolve overlapping intervals.
clock_diff_rows=[]
for metric,ylabel,name,lower in [('f1','Event-pair F1','results_clock_f1',.25),
                                ('gate_recall','True-pair gate recall','results_clock_gate',.35)]:
    fig,(ax,ad)=zoompair((3.62,3.36),(2.9,1.05))
    series={}
    for group,label,marker,style in [('clock_underreported',r'Declared: $0.8\,\mu$s','o','-'),
                                     ('clock_matched','Declared: matched','s','--')]:
        rr=S[(S.group==group)&(S.method=='joint')].copy()
        rr['x']=rr.value.astype(float);rr=rr.sort_values('x')
        x=rr.x.to_numpy();y=rr[metric].to_numpy()
        elo=y-rr[metric+'_lo'].to_numpy();ehi=rr[metric+'_hi'].to_numpy()-y
        ax.plot(x,y,linestyle=style,linewidth=1.35,label=label)
        shift=-.12 if group=='clock_underreported' else .12
        ax.errorbar(x+shift,y,yerr=[elo,ehi],fmt=marker,linestyle='none',
                    markersize=3.6,capsize=2.1,elinewidth=.85)
        series[group]=(x,y)
    ax.set(ylabel=ylabel,ylim=(lower,1.025),xlim=(-.7,20.7))
    ax.legend(loc='lower left',frameon=False,fontsize=7.3)
    x,a=series['clock_underreported'];xx,b=series['clock_matched']
    assert np.allclose(x,xx)
    delta=b-a
    ad.axhline(0,color='.43',linewidth=.8)
    ad.plot(x,delta,'D-',linewidth=1.1,markersize=3.2)
    ad.set(xlabel=r'Actual clock s.d. ($\mu$s)',ylabel=r'$\Delta$ matched-fixed',xlim=(-.7,20.7))
    ad.set_ylim(-.022,max(.045,float(np.max(delta))*1.20))
    ad.yaxis.set_major_locator(plt.MaxNLocator(3))
    save(fig,name)
    clock_diff_rows.extend({'metric':metric,'clock_s.d._microseconds':float(xx),'matched_minus_fixed':float(dd)}
                           for xx,dd in zip(x,delta))
S[S.group.isin(['clock_underreported','clock_matched'])].to_csv(D/'clock_shift.csv',index=False)
pd.DataFrame(clock_diff_rows).to_csv(D/'clock_difference.csv',index=False)

# Figure 11b: separate the close runtime trajectories into aligned *small
# multiples* using identical log axes. A third axis expands the real percent
# difference, rather than manufacturing a vertical gap in measured runtime.
fig, axs=plt.subplots(3,1,figsize=(3.62,3.36),sharex=True,
    gridspec_kw={'height_ratios':[1.55,1.55,1.08]})
fig.subplots_adjust(left=.19,right=.975,bottom=.15,top=.96,hspace=.09)
times={}
for ax,m,label,marker,style in [(axs[0],'physics','Physical','o','-'),
                                (axs[1],'joint','Joint linear','s','--')]:
    rr=S[(S.group=='scaling')&(S.method==m)].copy()
    rr['x']=rr.value.astype(float);rr=rr.sort_values('x')
    x=rr.observations.to_numpy();y=rr.runtime_ms.to_numpy()
    errlo=y-rr.runtime_ms_lo.to_numpy();errhi=rr.runtime_ms_hi.to_numpy()-y
    ax.errorbar(x,y,yerr=[errlo,errhi],fmt=marker,linestyle=style,
                capsize=2.4,markersize=4.0,linewidth=1.35,elinewidth=.75)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(25,245);ax.set_ylim(.19,11)
    ax.grid(alpha=.23,linewidth=.6)
    ax.text(.055,.92,label,transform=ax.transAxes,ha='left',va='top',fontsize=8.0)
    ax.tick_params(axis='y',labelsize=7.0)
    times[m]=(x,y)
axs[0].set_ylabel('Time (ms)',fontsize=8)
axs[1].set_ylabel('Time (ms)',fontsize=8)
x,p=times['physics'];xx,j=times['joint'];assert np.allclose(x,xx)
rel=100*(j/p-1.0)
axs[2].axhline(0,color='.43',linewidth=.8)
axs[2].plot(x,rel,'D-',markersize=3.25,linewidth=1.05)
axs[2].set(ylabel=r'$\Delta$ time (%)',xlabel='Mean observations per scene',xlim=(25,245))
axs[2].set_xscale('log');axs[2].set_xticks([30,50,100,200]);axs[2].set_xticklabels(['30','50','100','200'])
axs[2].xaxis.set_minor_formatter(plt.NullFormatter())
axs[2].grid(alpha=.23,linewidth=.6)
axs[2].set_ylim(min(-6.8,float(np.min(rel))-1.0),max(6.8,float(np.max(rel))+1.0))
axs[2].yaxis.set_major_locator(plt.MaxNLocator(3))
save(fig,'results_runtime')
S[S.group=='scaling'].to_csv(D/'runtime.csv',index=False)
pd.DataFrame({'mean_observations':x,'physical_ms':p,'joint_ms':j,
              'joint_minus_physical_percent':rel}).to_csv(D/'runtime_difference.csv',index=False)


# Additional diagnostics retained in the supplement, not removed from the study.
fig, ax = axpair(); riskrows=[]
for m,label,style in [('physics_calibrated','Physical','-'),('joint_calibrated','Joint linear','--'),
                      ('boosted_calibrated','Boosted joint','-.')]:
    a=np.load(R/f'prediction_{m}_r00.npz');y=a['y'][a['gate']];p=a['prob'][a['gate']]
    order=np.argsort(-np.maximum(p,1-p),kind='stable')
    risk=np.cumsum(((p>=.5)!=y)[order])/np.arange(1,len(y)+1)
    ns=np.maximum(np.unique(np.rint(np.linspace(.01,1,100)*len(y)).astype(int)),1)
    cov=ns/len(y);ax.plot(cov,risk[ns-1],label=label,linestyle=style,linewidth=1.4)
    riskrows.extend(dict(method=m,coverage=float(x),risk=float(z)) for x,z in zip(cov,risk[ns-1]))
ax.set(xlabel='Coverage of admitted pairs',ylabel='Retained binary error rate',xlim=(0,1))
ax.legend(loc='upper left',frameon=False);save(fig,'supp_risk_coverage')
pd.DataFrame(riskrows).to_csv(D/'risk_coverage.csv',index=False)

E=pd.read_csv(R/'exact_cases.csv')
fig, ax = axpair()
ax.bar(np.arange(1,len(E)+1), 100*E.relative_gap, width=.85)
ax.set(xlabel='Small audit instance (fixed order)',ylabel='Relative objective gap (%)',xlim=(0,len(E)+1))
save(fig, 'supp_exact_gap')
E.to_csv(D/'exact_objective_gap.csv',index=False)
summary = {'exact_cases':len(E),'exact_objective_matches':int((E.relative_gap.abs()<1e-8).sum()),
 'mean_exact_gap':float(E.relative_gap.mean()),'max_exact_gap':float(E.relative_gap.max()),
 'min_n':int(E.n.min()),'max_n':int(E.n.max()),
 'mean_greedy_f1':float(E.greedy_f1.mean()),'mean_exact_f1':float(E.exact_f1.mean())}
(ROOT/'audit'/'derived_summary.json').write_text(json.dumps(summary,indent=2))
print('Generated 9 statistical panels and 4 tables from executed outputs.')
print(json.dumps(summary,indent=2))
