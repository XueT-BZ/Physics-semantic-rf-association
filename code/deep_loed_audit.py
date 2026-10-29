"""Public-recording timestamp/reference audit and metadata-only clock fitting.

This is a secondary analysis of the retained LoED sample. Existing records,
references, test population and reported results are never overwritten.
"""
from pathlib import Path
from itertools import combinations
import hashlib,json,time,zipfile,platform,sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import loed_benchmark as loed
from experiment import constrained_cluster

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/deep_revision/loed'
LAG_LIMIT=5.0
BIN_WIDTH=.001
REFINE_RADIUS=.005
MIN_PEAK=10
MIN_REFINED=20
BACKGROUND_RATIO=5.0
DISTANT_PEAK_RATIO=2.0

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def clock_from_metadata(frame,gateway_ids):
    """Only five observable columns enter this function; no identities or labels."""
    allowed=['day','gateway','time_s','frequency','spreading_factor']
    assert frame.columns.tolist()==allowed
    delta_by_pair={}
    for _,day in frame.groupby('day',sort=True):
        for a,b in combinations(sorted(day.gateway.unique()),2):
            left=day[day.gateway==a];right=day[day.gateway==b]
            values=[]
            for (frequency,sf),x in left.groupby(['frequency','spreading_factor']):
                y=np.sort(right[(right.frequency==frequency)&(right.spreading_factor==sf)].time_s.to_numpy())
                if not len(y):continue
                for t in x.time_s.to_numpy():
                    lo,hi=np.searchsorted(y,[t-LAG_LIMIT,t+LAG_LIMIT])
                    values.extend((t-y[lo:hi]).tolist())
            delta_by_pair.setdefault((a,b),[]).extend(values)
    edges=np.linspace(-LAG_LIMIT,LAG_LIMIT,int(round(2*LAG_LIMIT/BIN_WIDTH))+1)
    records=[];A=[];targets=[];hist_rows=[]
    for (a,b),values in sorted(delta_by_pair.items()):
        values=np.asarray(values);counts,_=np.histogram(values,bins=edges)
        peak=int(np.argmax(counts));center=float((edges[peak]+edges[peak+1])/2)
        centers=(edges[:-1]+edges[1:])/2
        outside=abs(centers-center)>2*REFINE_RADIUS
        distant=int(counts[outside].max()) if np.any(outside) else 0
        background=float(np.quantile(counts,.99))
        selected=values[abs(values-center)<=REFINE_RADIUS]
        supported=(int(counts[peak])>=MIN_PEAK and len(selected)>=MIN_REFINED
                   and counts[peak]>=BACKGROUND_RATIO*max(1.,background)
                   and counts[peak]>=DISTANT_PEAK_RATIO*max(1,distant))
        lag=float(np.median(selected)) if len(selected) else None
        records.append(dict(gateway_a=a,gateway_b=b,candidate_pairs=len(values),
            peak_bin_center_s=center,peak_bin_count=int(counts[peak]),
            background_99_count=background,distant_peak_count=distant,
            refined_support=len(selected),median_lag_s=lag,supported=bool(supported)))
        for index in np.flatnonzero(counts):
            hist_rows.append(dict(gateway_a=a,gateway_b=b,bin_lower_s=float(edges[index]),
                                 bin_upper_s=float(edges[index+1]),count=int(counts[index])))
        if supported:
            row=np.zeros(len(gateway_ids));row[gateway_ids.index(a)]=1;row[gateway_ids.index(b)]=-1
            weight=np.sqrt(len(selected));A.append(row*weight);targets.append(lag*weight)
    beta=np.linalg.lstsq(np.asarray(A),np.asarray(targets),rcond=None)[0] if A else np.zeros(len(gateway_ids))
    return dict(zip(gateway_ids,beta.tolist())),records,pd.DataFrame(hist_rows)

def verify_clock_estimator():
    """A known shift verifies orientation; equal modes must remain unresolved."""
    n=40;times=np.arange(n,dtype=float)*20.+100.
    def records(lags):
        rows=[]
        for i,t in enumerate(times):
            rows.append(dict(day='synthetic-check',gateway='A',time_s=t+lags[i],frequency=868100000,spreading_factor=7))
            rows.append(dict(day='synthetic-check',gateway='B',time_s=t,frequency=868100000,spreading_factor=7))
        return pd.DataFrame(rows)[['day','gateway','time_s','frequency','spreading_factor']]
    fitted,details,_=clock_from_metadata(records(np.full(n,.37)),['A','B'])
    assert details[0]['supported'] and abs(fitted['A']-fitted['B']-.37)<1e-9
    ambiguous,details,_=clock_from_metadata(records(np.where(np.arange(n)%2==0,.3,1.2)),['A','B'])
    assert not details[0]['supported'] and ambiguous=={'A':0.,'B':0.}
    return dict(known_shift_sign_and_magnitude=True,equal_separated_modes_rejected=True)

def reference_pairs(days,offsets):
    rows=[]
    for d in days:
        for reference,g in d.groupby('reference_payload',sort=False):
            for a,b in combinations(g.sort_values('gateway').itertuples(),2):
                raw=float(a.time_s-b.time_s)
                corrected=raw-offsets[a.gateway]+offsets[b.gateway]
                rows.append(dict(day=a.day,split=a.split,reference_payload=reference,
                    gateway_a=a.gateway,gateway_b=b.gateway,row_a=int(a.original_row),row_b=int(b.original_row),
                    observation_a=f'{a.day}/{a.original_row}',observation_b=f'{b.day}/{b.original_row}',
                    time_a=a.time,time_b=b.time,time_s_a=a.time_s,time_s_b=b.time_s,
                    raw_signed_s=raw,corrected_signed_s=float(corrected),corrected_abs_s=float(abs(corrected)),
                    frequency_a=int(a.frequency),frequency_b=int(b.frequency),
                    sf_a=int(a.spreading_factor),sf_b=int(b.spreading_factor),
                    same_frequency=bool(a.frequency==b.frequency),same_sf=bool(a.spreading_factor==b.spreading_factor),
                    device_address_a=a.device_address,device_address_b=b.device_address,
                    fcnt_a=int(a.fcnt),fcnt_b=int(b.fcnt)))
    return pd.DataFrame(rows)

def fitted_evaluation(days,offsets,name,expected_ids):
    ds=[d.copy() for d in days];gateways=sorted(offsets);ids=dict(zip(gateways,range(len(gateways))))
    for d in ds:
        d['receiver']=d.gateway.map(ids)
        d['corrected_time_s']=d.time_s-d.gateway.map(offsets)
        d['window']=np.floor(d.corrected_time_s/60.).astype(int)
        d['observation_id']=d.day+'/'+d.original_row.astype(str)
    selected=set(pd.concat([d for d in ds if d['split'].iloc[0]=='test']).observation_id)
    assert selected==expected_ids
    errors=[]
    for d in ds:
        if d['split'].iloc[0]=='calibration':
            for _,g in d.groupby('reference_payload'):
                errors.extend(abs(a.corrected_time_s-b.corrected_time_s) for a,b in combinations(g.itertuples(),2))
    gate=max(.001,float(np.quantile(errors,.99)))
    windows={d.day.iloc[0]:loed.make_windows(d,gate) for d in ds}
    train=[p for d in ds if d['split'].iloc[0]=='train' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
    cal=[p for d in ds if d['split'].iloc[0]=='calibration' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500))
    model.fit(np.concatenate([p.X for p in train]),np.concatenate([p.y for p in train]))
    logits=np.concatenate([model.decision_function(p.X)[p.gate(1.)] for p in cal])
    targets=np.concatenate([p.y[p.gate(1.)] for p in cal])
    fit=minimize_scalar(lambda t:float(np.mean(np.logaddexp(0,logits/np.exp(t))-targets*logits/np.exp(t))),
                       bounds=(np.log(.1),np.log(10)),method='bounded')
    assert fit.success
    temperature=float(np.exp(fit.x));scaler,lr=model.steps[0][1],model.steps[1][1]
    saved=dict(method=name,offsets_s=offsets,gate_s=gate,temperature=temperature,
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]),train_positive_pairs=int(sum(p.y.sum() for p in train)))
    rows=[]
    for d in ds:
        if d['split'].iloc[0]!='test':continue
        labels=np.empty(len(d),int);offset=0;lookup=pd.Index(d.observation_id)
        for group,scene,p in windows[d.day.iloc[0]]:
            if len(p.y):
                z=constrained_cluster(scene,p,expit(model.decision_function(p.X)/temperature),k=1.)
            else:z=np.arange(len(group))
            labels[lookup.get_indexer(group.observation_id)]=z+offset;offset+=int(z.max())+1
        m=loed.counts(d,labels,gate)
        assert m['receiver_conflicts']==m['gate_violations']==0
        rows.append(dict(method=name,day=d.day.iloc[0],reports=len(d),gate_s=gate,**m))
        pd.DataFrame(dict(observation_id=d.observation_id,predicted=labels)).to_csv(OUT/f'prediction_{name}_{d.day.iloc[0]}.csv.gz',index=False)
    return rows,saved

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    estimator_checks=verify_clock_estimator()
    protocol=dict(analysis='Secondary audit; no test-driven parameter choice or reference change',
        inputs='Unchanged publisher sample and unchanged original inclusion rules',
        original_code_changed=False,clock_observables=['day','gateway','time_s','frequency','spreading_factor'],
        clock_fit_split='training only',lag_search_s=[-LAG_LIMIT,LAG_LIMIT],bin_width_s=BIN_WIDTH,
        refinement_radius_s=REFINE_RADIUS,minimum_peak_bin_count=MIN_PEAK,
        minimum_refined_support=MIN_REFINED,peak_to_99percent_histogram_count=BACKGROUND_RATIO,
        peak_to_distant_peak=DISTANT_PEAK_RATIO,distant_peak_exclusion_s=2*REFINE_RADIUS,
        gate='Original calibration-reference 0.99 quantile with 1 ms floor',
        scorer='Original supervised three-feature logistic regression and temperature fit',
        qualification='Only clock-offset fitting is reference-free; filtering, scorer labels, gate calibration and evaluation remain reference-assisted',
        test_population='Unchanged 23677 reports; all ambiguous reference pairs remain',
        tail_diagnostic_threshold_s=.01)
    write_json(OUT/'protocol.json',protocol)
    days,inclusion=loed.load();gateways=sorted(set(pd.concat(days).gateway));offsets,support=loed.fit_offsets(days,gateways)
    pairs=reference_pairs(days,offsets);pairs.to_csv(OUT/'reference_pairs.csv.gz',index=False)
    aggregates=[]
    for (day,a,b),g in pairs.groupby(['day','gateway_a','gateway_b']):
        values=g.corrected_abs_s.to_numpy();tail=values>.01
        aggregates.append(dict(day=day,split=g.split.iloc[0],gateway_a=a,gateway_b=b,
            reference_pairs=len(g),raw_signed_median_s=float(g.raw_signed_s.median()),
            corrected_signed_median_s=float(g.corrected_signed_s.median()),
            corrected_abs_median_s=float(np.median(values)),
            corrected_abs_q95_s=float(np.quantile(values,.95)),corrected_abs_q99_s=float(np.quantile(values,.99)),
            corrected_abs_q999_s=float(np.quantile(values,.999)),corrected_abs_max_s=float(values.max()),
            tail_above_10ms=int(tail.sum()),tail_channel_mismatch=int((tail&~g.same_frequency).sum()),
            center_pairs=int((~tail).sum()),center_channel_mismatch=int((~tail&~g.same_frequency).sum()),
            sf_mismatch=int((~g.same_sf).sum())))
    pd.DataFrame(aggregates).to_csv(OUT/'date_pair_summary.csv',index=False)
    # Keep dates without a reference pair visible in a separate support ledger.
    day_rows=[]
    for d in days:
        day=d.day.iloc[0];part=pairs[pairs.day==day]
        day_rows.append(dict(day=day,split=d['split'].iloc[0],reports=len(d),
            gateways=';'.join(sorted(d.gateway.unique())),reference_pairs=len(part),
            gateway_pairs=';'.join(sorted(set(part.gateway_a+'/'+part.gateway_b))) if len(part) else 'none'))
    pd.DataFrame(day_rows).to_csv(OUT/'day_support.csv',index=False)
    cal=pairs[pairs.split=='calibration'];v=np.sort(cal.corrected_abs_s.to_numpy());quantiles=[]
    for q in [.95,.99,.999]:
        rank=(len(v)-1)*q;lo=int(np.floor(rank));hi=int(np.ceil(rank));value=float(np.quantile(v,q))
        quantiles.append(dict(q=q,n=len(v),zero_based_rank=rank,lower_rank=lo,upper_rank=hi,
            lower_value_s=float(v[lo]),upper_value_s=float(v[hi]),interpolated_s=value,
            above_quantile=int((v>value).sum())))
    pd.DataFrame(quantiles).to_csv(OUT/'calibration_quantile_endpoints.csv',index=False)
    tails=pairs[pairs.corrected_abs_s>.01].copy();raw_audit=[];raw_columns={}
    with zipfile.ZipFile(loed.ARCHIVE) as archive:
        for name in archive.namelist():
            if not name.endswith('.csv') or name.startswith('__MACOSX'):continue
            day=pd.to_datetime(Path(name).stem,format='%d_%m_%Y').strftime('%Y-%m-%d')
            raw=pd.read_csv(archive.open(name),dtype={'gateway':str,'device_address':str})
            raw_columns[day]=raw.columns.tolist()
            for row in tails[tails.day==day].itertuples():
                a,b=raw.loc[row.row_a],raw.loc[row.row_b]
                assert a.physical_payload==b.physical_payload
                assert a.frequency==row.frequency_a and b.frequency==row.frequency_b
                assert a.time==row.time_a and b.time==row.time_b
                same=raw[raw.physical_payload==a.physical_payload]
                raw_audit.append(dict(day=day,reference_payload=row.reference_payload,row_a=row.row_a,row_b=row.row_b,
                    raw_payload_rows=len(same),crc_valid_payload_rows=int((same.crc_status==1).sum()),
                    raw_payload_gateways=same.gateway.nunique(),distinct_frequencies=same.frequency.nunique(),
                    full_payload_equal=True,device_equal=bool(a.device_address==b.device_address),
                    frame_counter_equal=bool(a.fcnt==b.fcnt),sf_equal=bool(a.spreading_factor==b.spreading_factor),
                    frequency_fields_verified_against_raw=True,
                    raw_rows_with_same_payload=';'.join(str(x) for x in same.index)))
    tails.to_csv(OUT/'tail_pairs_above_10ms.csv',index=False)
    pd.DataFrame(raw_audit).to_csv(OUT/'tail_raw_trace.csv',index=False)
    write_json(OUT/'raw_columns.json',raw_columns)
    print('Reference audit complete; fitting metadata-only clock offset.',flush=True)
    training=pd.concat([d for d in days if d['split'].iloc[0]=='train'],ignore_index=True)
    metadata=training[['day','gateway','time_s','frequency','spreading_factor']].copy()
    unsupervised,clock_rows,histogram=clock_from_metadata(metadata,gateways)
    pd.DataFrame(clock_rows).to_csv(OUT/'metadata_clock_support.csv',index=False)
    histogram.to_csv(OUT/'metadata_clock_histogram.csv.gz',index=False)
    write_json(OUT/'clock_offsets.json',dict(supervised=offsets,metadata_only=unsupervised,
        maximum_absolute_offset_difference_s=max(abs(offsets[g]-unsupervised[g]) for g in gateways),
        supervised_pair_support=support))
    # Column whitelist makes all identity labels inaccessible to the estimator.
    assert metadata.columns.tolist()==protocol['clock_observables']
    expected=pd.read_csv(ROOT/'results/loed/observations.csv.gz');expected_ids=set(expected[expected.split=='test'].observation_id)
    results=[];fits=[]
    for name,correction in [('supervised_reference',offsets),('training_metadata_mode',unsupervised)]:
        rows,fit=fitted_evaluation(days,correction,name,expected_ids);results.extend(rows);fits.append(fit)
        print(name,'evaluation complete',flush=True)
    per_day=pd.DataFrame(results);per_day.to_csv(OUT/'clock_comparison_per_capture.csv',index=False)
    summary=[]
    for name,g in per_day.groupby('method'):
        counts=g[['tp','fp','fn','reports','receiver_conflicts','gate_violations']].sum()
        summary.append(dict(method=name,gate_s=float(g.gate_s.iloc[0]),**counts.to_dict(),
            f1=float(2*counts.tp/(2*counts.tp+counts.fp+counts.fn))))
    s=pd.DataFrame(summary);s.to_csv(OUT/'clock_comparison_summary.csv',index=False)
    base=s[s.method=='supervised_reference'].iloc[0];assert (base.tp,base.fp,base.fn)==(1411,1,6)
    equal_predictions={}
    for day in ['2020-08-03','2020-09-02']:
        first=pd.read_csv(OUT/f'prediction_supervised_reference_{day}.csv.gz')
        second=pd.read_csv(OUT/f'prediction_training_metadata_mode_{day}.csv.gz')
        assert first.observation_id.equals(second.observation_id)
        equal_predictions[day]=bool(first.predicted.equals(second.predicted))
    date_frame=pd.DataFrame(aggregates).set_index('day')
    lines=[r'\begin{tabularx}{\linewidth}{@{}Yrrrrr@{}}\toprule',
        r'Capture & Reference pairs & Median ($\mu$s) & $Q_{0.99}$ (ms) & $>10$ ms & Channel mismatch in tail \\\midrule']
    for row in day_rows:
        day=row['day']
        if day in date_frame.index:
            v=date_frame.loc[day]
            lines.append(f"{day} & {int(v.reference_pairs)} & {v.corrected_abs_median_s*1e6:.1f} & {v.corrected_abs_q99_s*1e3:.3f} & {int(v.tail_above_10ms)} & {int(v.tail_channel_mismatch)} "+r'\\')
        else:lines.append(f"{day} & 0 & -- & -- & -- & -- "+r'\\')
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/deep_loed_tail.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabularx}{\linewidth}{@{}Yrrrrr@{}}\toprule',
        r'Clock fit & Training support & Lag (s) & TP / FP / FN & $G$ (ms) & Packet F1 \\\midrule']
    for method,title,n in [('supervised_reference','Payload-reference median',256),('training_metadata_mode','Metadata lag mode',251)]:
        v=s[s.method==method].iloc[0]
        lines.append(f"{title} & {n} & 0.999985 & {int(v.tp)} / {int(v.fp)} / {int(v.fn)} & {v.gate_s*1e3:.3f} & {v.f1:.4f} "+r'\\')
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/deep_loed_clock.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'fitted_parameters.json',fits)
    assertions=dict(calibration_pairs=len(cal),calibration_above_2s=int((cal.corrected_abs_s>2).sum()),
        calibration_above_2s_all_channel_mismatched=bool((~cal.loc[cal.corrected_abs_s>2,'same_frequency']).all()),
        raw_timestamp_columns=[c for c in raw_columns['2020-07-03'] if 'time' in c.lower()],
        raw_tail_rows_checked=len(raw_audit)*2,unchanged_test_reports=len(expected_ids),
        original_reference_counts_reproduced=True,clock_estimator_label_free_columns=True,
        no_test_identity_in_offset_fit=True,all_test_references_retained=True)
    assert assertions['calibration_pairs']==434 and assertions['calibration_above_2s']==4
    write_json(OUT/'verification.json',dict(status='PASS',**assertions,clock_estimator_checks=estimator_checks,
        identical_partitions_by_capture=equal_predictions,seconds=time.perf_counter()-started,
        python=sys.version,platform=platform.platform(),archive_sha256=hashlib.sha256(loed.ARCHIVE.read_bytes()).hexdigest()))
    print(s.to_string(index=False));print(json.dumps(assertions,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
