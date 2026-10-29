"""Training-block and calibration-block sensitivity on unchanged public test captures."""
from itertools import combinations
import json,time,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import loed_benchmark as loed
from experiment import constrained_cluster
from review_common import ROOT,OUT,environment

DEST=OUT/'public_sensitivity'

def offsets_with_support(days,gateways):
    counts={}
    for d in days:
        if d['split'].iloc[0]!='train':continue
        for _,g in d.groupby('reference_payload'):
            for a,b in combinations(sorted(g.gateway),2):counts[(a,b)]=counts.get((a,b),0)+1
    support=[dict(gateway_a=a,gateway_b=b,reference_pairs=n,supported=n>=10) for (a,b),n in sorted(counts.items())]
    if any(row['supported'] for row in support):
        offsets,rows=loed.fit_offsets(days,gateways)
    else:
        offsets={g:0. for g in gateways};rows=[]
    return offsets,rows,support

def main():
    DEST.mkdir(parents=True,exist_ok=True);environment('public');started=time.perf_counter()
    original_days,inclusion=loed.load()
    inclusion.to_csv(DEST/'fixed_original_inclusion.csv',index=False)
    by_day={d.day.iloc[0]:d for d in original_days}
    dates=sorted(by_day);pretest=dates[:4];test_days=dates[4:]
    settings=[
        ('original_split',pretest[:3],pretest[3],.99,True),
        ('first_two_train',pretest[:2],pretest[3],.99,True),
        ('latest_train_only',[pretest[2]],pretest[3],.99,True),
        ('second_train_only',[pretest[1]],pretest[3],.99,True),
        ('early_calibration',[pretest[0]],pretest[1],.99,True),
        ('middle_calibration',pretest[:2],pretest[2],.99,True),
        ('uncorrected_clocks',pretest[:3],pretest[3],.99,False),
        ('gate_quantile_095',pretest[:3],pretest[3],.95,True),
        ('gate_quantile_0999',pretest[:3],pretest[3],.999,True)]
    protocol=dict(settings=[dict(name=a,training=b,calibration=c,quantile=q,clock_correction=f) for a,b,c,q,f in settings],
                  test_captures=test_days,window_seconds=60.,test_selection_used_for_parameter_fitting=False,
                  no_synthetic_noise_or_posteriors=True,claim='Preprocessing sensitivity only; no multi-protocol semantic validation.')
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    gateways=sorted(set(pd.concat(original_days).gateway));receiver_ids=dict(zip(gateways,range(len(gateways))))
    released=pd.read_csv(ROOT/'results/loed/observations.csv.gz')
    expected_ids=set(released[released.split=='test'].observation_id)
    rows=[];fit_records=[];exclusions=[]
    for number,(name,training,calibration,quantile,correct_clocks) in enumerate(settings):
        days=[]
        for day in training+[calibration]+test_days:
            d=by_day[day].copy();d['split']='train' if day in training else 'calibration' if day==calibration else 'test'
            days.append(d)
        membership=pd.concat([d[['reference_payload','split']] for d in days]).drop_duplicates()
        shared=membership.groupby('reference_payload').split.nunique();shared=set(shared[shared>1].index)
        for i,d in enumerate(days):
            keep=~d.reference_payload.isin(shared)
            exclusions.append(dict(setting=name,day=d.day.iloc[0],split=d['split'].iloc[0],additional_cross_split_rows_removed=int((~keep).sum())))
            days[i]=d[keep].copy().reset_index(drop=True)
        offsets,offset_rows,offset_support=offsets_with_support(days,gateways)
        if not correct_clocks:offsets={g:0. for g in gateways}
        for d in days:
            d['receiver']=d.gateway.map(receiver_ids)
            d['corrected_time_s']=d.time_s-d.gateway.map(offsets)
            d['window']=np.floor(d.corrected_time_s/60.).astype(int)
            d['observation_id']=d.day+'/'+d.original_row.astype(str)
        observed=set(pd.concat([d for d in days if d['split'].iloc[0]=='test']).observation_id)
        assert observed==expected_ids, 'Public test population changed'
        errors=[]
        for d in days:
            if d['split'].iloc[0]=='calibration':
                for _,group in d.groupby('reference_payload'):
                    errors.extend(abs(a.corrected_time_s-b.corrected_time_s) for a,b in combinations(group.itertuples(),2))
        if not errors:
            fit_records.append(dict(setting=name,status='insufficient_calibration_matches',training=training,calibration=calibration,clock_support=offset_support))
            print(f'Public preprocessing {number+1}/{len(settings)}: {name}; no calibration matches',flush=True)
            continue
        gate=max(.001,float(np.quantile(errors,quantile)))
        windows={d.day.iloc[0]:loed.make_windows(d,gate) for d in days}
        train=[p for d in days if d['split'].iloc[0]=='train' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
        cal=[p for d in days if d['split'].iloc[0]=='calibration' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
        record=dict(setting=name,status='evaluated',training=training,calibration=calibration,gate_quantile=quantile,
                    clock_correction=correct_clocks,offsets_s=offsets,clock_training_pairs=offset_rows,
                    clock_support=offset_support,calibration_reference_pairs=len(errors),gate_s=gate,models={})
        train_targets=np.concatenate([p.y for p in train]) if train else np.array([])
        record['training_positive_pairs']=int(train_targets.sum())
        record['training_pairs']=len(train_targets)
        if len(np.unique(train_targets))<2:
            record['status']='insufficient_training_classes';fit_records.append(record)
            print(f'Public preprocessing {number+1}/{len(settings)}: {name}; training has fewer than two classes',flush=True)
            continue
        for scorer,cols in [('timing',[0]),('metadata',[0,1,2])]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500))
            model.fit(np.concatenate([p.X[:,cols] for p in train]),np.concatenate([p.y for p in train]))
            logits=np.concatenate([model.decision_function(p.X[:,cols])[p.gate(1.)] for p in cal])
            targets=np.concatenate([p.y[p.gate(1.)] for p in cal])
            fitted=minimize_scalar(lambda t:float(np.mean(np.logaddexp(0,logits/np.exp(t))-targets*logits/np.exp(t))),
                                   bounds=(np.log(.1),np.log(10)),method='bounded')
            assert fitted.success
            temperature=float(np.exp(fitted.x));scaler,lr=model.steps[0][1],model.steps[1][1]
            record['models'][scorer]=dict(columns=cols,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficient=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),temperature=temperature)
            for d in days:
                if d['split'].iloc[0]!='test':continue
                day=d.day.iloc[0];labels=np.empty(len(d),int);offset=0;lookup=pd.Index(d.observation_id)
                for group,scene,p in windows[day]:
                    if len(p.y):
                        probability=expit(model.decision_function(p.X[:,cols])/temperature)
                        z=constrained_cluster(scene,p,probability,k=1.)
                    else:z=np.arange(len(group))
                    locations=lookup.get_indexer(group.observation_id)
                    labels[locations]=z+offset;offset+=int(z.max())+1
                m=loed.counts(d,labels,gate)
                assert m['receiver_conflicts']==m['gate_violations']==0
                rows.append(dict(setting=name,scorer=scorer,day=day,reports=len(d),gate_s=gate,**m))
                pd.DataFrame(dict(observation_id=d.observation_id,predicted=labels)).to_csv(DEST/f'prediction_{name}_{scorer}_{day}.csv.gz',index=False)
        fit_records.append(record)
        print(f'Public preprocessing {number+1}/{len(settings)}: {name}; {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'per_capture.csv',index=False)
    pooled=[]
    for (name,scorer),g in frame.groupby(['setting','scorer']):
        sums=g[['tp','fp','fn','reports','receiver_conflicts','gate_violations']].sum()
        pooled.append(dict(setting=name,scorer=scorer,gate_s=float(g.gate_s.iloc[0]),**sums.to_dict(),
                           f1=float(2*sums.tp/max(1,2*sums.tp+sums.fp+sums.fn))))
    summary=pd.DataFrame(pooled);summary.to_csv(DEST/'summary.csv',index=False)
    baseline=summary[(summary.setting=='original_split')&(summary.scorer=='metadata')].iloc[0]
    assert (baseline.tp,baseline.fp,baseline.fn)==(1411,1,6),baseline.to_dict()
    pd.DataFrame(exclusions).to_csv(DEST/'additional_split_exclusions.csv',index=False)
    (DEST/'fitted_parameters.json').write_text(json.dumps(fit_records,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'models','offsets_s','clock_training_pairs','clock_support'}} for r in fit_records]).to_csv(DEST/'setting_status.csv',index=False)
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-started,
        settings=len(settings),evaluated_settings=frame.setting.nunique(),test_reports_unchanged=len(expected_ids),
        original_metadata_counts_reproduced=True,archive_md5=hashlib.md5(loed.ARCHIVE.read_bytes()).hexdigest()),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
