"""Render the semantic stress figure and descriptive dependence checks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deep_semantic_stress import CONDITIONS, SCORERS, TITLES, ROOT, DEST


def main():
    data = pd.read_csv(DEST/'summary.csv').set_index(['condition','scorer'])
    names = ['Independent classes', 'Same-class near collisions',
             'Dense collisions: independent classes', 'Dense collisions: same class',
             'Class imbalance', 'Receiver-specific classifiers',
             'Independent class errors', 'Event-shared class errors']
    colors = ['#687787','#1762A5','#CE721A','#2F8465']
    labels = ['Physical','Weighted linear','Unweighted linear','Boosted']
    fig, ax = plt.subplots(figsize=(9.2,5.3))
    y = np.arange(len(CONDITIONS))[::-1]
    for i,(scorer,color,label) in enumerate(zip(SCORERS,colors,labels)):
        part = data.loc[[(c,scorer) for c in CONDITIONS]]
        means = part.f1.to_numpy()
        error = np.vstack([means-part.f1_lo.to_numpy(),part.f1_hi.to_numpy()-means])
        ax.errorbar(means,y+(i-1.5)*.16,xerr=error,fmt='o',markersize=4.2,
                    capsize=2.5,lw=1.05,color=color,label=label)
    ax.set_yticks(y,names,fontsize=9)
    ax.set_xlabel('Event-pair F1',fontsize=10)
    ax.set_xlim(.39,.95)
    ax.set_ylim(-.6,len(CONDITIONS)-.4)
    ax.grid(axis='x',color='#DDE2E6',lw=.7)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=4,frameon=False,
              columnspacing=1.25,handletextpad=.4,fontsize=8.6)
    fig.subplots_adjust(left=.335,right=.98,bottom=.115,top=.88)
    fig.savefig(ROOT/'figures/deep_semantic_stress.pdf')
    fig.savefig(DEST/'deep_semantic_stress.png',dpi=160)
    plt.close(fig)
    audit=pd.read_csv(DEST/'classification_audit.csv.gz')
    output=[]
    for c in ['independent_errors_control','event_shared_errors']:
        t=audit[(audit.condition==c)&(~audit.missing)&(audit.event>=0)]
        xx=[]; yy=[]; disagree=[]
        for _,g in t.groupby(['replication','scene','event']):
            ii,jj=np.triu_indices(len(g),1)
            w=g.wrong.to_numpy(float); q=g.predicted_class.to_numpy()
            xx.extend(w[ii]); yy.extend(w[jj]); disagree.extend(q[ii]!=q[jj])
        output.append(dict(condition=c,genuine_nonmissing_reports=len(t),
            error_fraction=float(t.wrong.mean()),mean_confidence=float(t.confidence.mean()),
            wrong_mean_confidence=float(t[t.wrong].confidence.mean()),
            correct_mean_confidence=float(t[~t.wrong].confidence.mean()),
            same_event_nonmissing_pairs=len(xx),
            pair_error_correlation=float(np.corrcoef(xx,yy)[0,1]),
            pair_class_disagreement=float(np.mean(disagree))))
    pd.DataFrame(output).to_csv(DEST/'error_dependence_audit.csv',index=False)
    print('Rendered semantic stress figure and regenerated dependence audit.')


if __name__=='__main__': main()
