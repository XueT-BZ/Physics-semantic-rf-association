"""Measured LoED report association with identity fields confined to evaluation.

Reference identities are unique, CRC-valid, unconfirmed-uplink full PHY payloads.
They are not independently instrumented physical-transmission truth. Repeated
payload/gateway groups and groups spanning >5 s are excluded, with counts saved.
The six publisher-selected days are split chronologically 3/1/2 before pairing.
Reported mode metadata is used directly; no synthetic protocol posteriors,
positions, measurement variances, or RF waveforms are supplied.
"""
from pathlib import Path
from itertools import combinations
import base64
import hashlib
import json
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from experiment import Scene, Pairs, constrained_cluster, connected_cluster

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/loed'
ARCHIVE = ROOT/'data/public/loed/LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip'
WINDOW = 60.


def payload_id(value):
    try:
        data = base64.b64decode(value, validate=True)
        if len(data) < 12 or data[0] >> 5 != 2:
            return None
        return hashlib.sha256(data).hexdigest()
    except (ValueError, TypeError):
        return None


def load():
    assert hashlib.md5(ARCHIVE.read_bytes()).hexdigest() == '7370baddc9b546354c26a71cd38981bc'
    days = []
    audit = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX')]
        names.sort(key=lambda n: pd.to_datetime(Path(n).stem, format='%d_%m_%Y'))
        for index, name in enumerate(names):
            day = pd.to_datetime(Path(name).stem, format='%d_%m_%Y').strftime('%Y-%m-%d')
            split = 'train' if index < 3 else ('calibration' if index == 3 else 'test')
            d = pd.read_csv(archive.open(name), dtype={'gateway': str, 'device_address': str})
            info = dict(day=day, split=split, raw_rows=len(d), raw_gateways=d.gateway.nunique())
            d = d[d.crc_status == 1].copy()
            info['crc_valid_rows'] = len(d)
            d['reference_payload'] = d.physical_payload.map(payload_id)
            d = d[d.reference_payload.notna()].copy()
            info['valid_unconfirmed_uplink_rows'] = len(d)
            # Exact duplicate input rows are log duplicates, not additional reports.
            before = len(d)
            d = d.drop_duplicates()
            info['exact_duplicate_rows_removed'] = before-len(d)
            stamp = pd.to_datetime(d.time, format='ISO8601', utc=True).dt.as_unit('ns')
            start = pd.Timestamp(day, tz='UTC').value
            d['time_s'] = (stamp.astype('int64')-start)/1e9
            # Some publisher files contain records just before the named day.
            # Keep their measured times; filename denotes the capture block.
            assert d.time_s.between(-86400, 172800).all(), 'Timestamp unit error'
            info['minimum_time_relative_to_named_day_s'] = float(d.time_s.min())
            info['maximum_time_relative_to_named_day_s'] = float(d.time_s.max())
            fields = d[['time_s','frequency','spreading_factor','bandwidth']].apply(pd.to_numeric, errors='coerce')
            valid = fields.notna().all(axis=1)&(fields.frequency > 0)&fields.spreading_factor.between(7,12)&(fields.bandwidth > 0)
            info['invalid_observable_rows_removed'] = int((~valid).sum())
            d = d[valid].copy()
            g = d.groupby('reference_payload').agg(n=('gateway','size'), gateways=('gateway','nunique'),
                                                    lo=('time_s','min'), hi=('time_s','max'))
            eligible = g[(g.n == g.gateways)&((g.hi-g.lo) <= 5)]
            info['repeated_gateway_or_long_span_groups_removed'] = len(g)-len(eligible)
            info['ambiguous_rows_removed'] = len(d)-int(eligible.n.sum())
            d = d[d.reference_payload.isin(eligible.index)].copy()
            d['day'] = day
            d['split'] = split
            d['original_row'] = d.index
            days.append(d)
            audit.append(info)
    # Remove payloads shared between distinct splits, independently of their score.
    membership = pd.concat([d[['reference_payload','split']] for d in days]).drop_duplicates()
    shared = membership.groupby('reference_payload').split.nunique()
    shared = set(shared[shared > 1].index)
    for index, d in enumerate(days):
        keep = ~d.reference_payload.isin(shared)
        audit[index]['cross_split_shared_payload_rows_removed'] = int((~keep).sum())
        d = d[keep].copy().reset_index(drop=True)
        g = d.groupby('reference_payload').gateway.nunique()
        audit[index].update(eligible_reports=len(d), reference_packets=len(g),
                            visible_reference_packets=int((g >= 2).sum()),
                            reference_positive_pairs=int(sum(n*(n-1)//2 for n in g)),
                            eligible_gateways=d.gateway.nunique())
        days[index] = d
    return days, pd.DataFrame(audit)


def fit_offsets(days, gateway_ids):
    values = {}
    for d in days:
        if d['split'].iloc[0] != 'train':
            continue
        for _, g in d.groupby('reference_payload'):
            for a,b in combinations(g.itertuples(), 2):
                if a.gateway < b.gateway:
                    key, delta = (a.gateway,b.gateway), a.time_s-b.time_s
                else:
                    key, delta = (b.gateway,a.gateway), b.time_s-a.time_s
                values.setdefault(key, []).append(delta)
    A, target, rows = [], [], []
    for (a,b), v in sorted(values.items()):
        if len(v) < 10:
            continue
        row = np.zeros(len(gateway_ids))
        row[gateway_ids.index(a)], row[gateway_ids.index(b)] = 1, -1
        weight = np.sqrt(len(v))
        A.append(row*weight)
        median = float(np.median(v))
        target.append(median*weight)
        rows.append(dict(gateway_a=a, gateway_b=b, training_pairs=len(v), median_difference_s=median))
    beta = np.linalg.lstsq(np.array(A), np.array(target), rcond=None)[0]
    return dict(zip(gateway_ids,beta)), rows


def pair_arrays(d):
    i,j = np.triu_indices(len(d), 1)
    receiver = d.receiver.to_numpy()
    keep = receiver[i] != receiver[j]
    return i[keep],j[keep]


def make_windows(d, gate):
    windows = []
    for window, g in d.groupby('window', sort=True):
        g = g.reset_index(drop=True)
        i,j = pair_arrays(g)
        delta = np.abs(g.corrected_time_s.to_numpy()[i]-g.corrected_time_s.to_numpy()[j])
        freq = g.frequency.to_numpy()
        sf = g.spreading_factor.to_numpy()
        X = np.column_stack([np.log1p(delta/gate), (freq[i] != freq[j]).astype(float),
                             (sf[i] != sf[j]).astype(float)])
        y = (g.reference_payload.to_numpy()[i] == g.reference_payload.to_numpy()[j]).astype(int)
        p = Pairs(X,X,X,y,i,j,delta/gate,np.zeros(len(i)),np.zeros(len(i),bool))
        truth = pd.factorize(g.reference_payload)[0]
        scene = Scene({'receiver':g.receiver.to_numpy()}, truth, np.zeros(len(g),bool),
                      np.empty((0,2)), int(window))
        windows.append((g,scene,p))
    return windows


def counts(d, z, gate):
    frame = pd.DataFrame(dict(reference=d.reference_payload.to_numpy(), z=z, receiver=d.receiver.to_numpy()))
    comb = lambda a: int(np.sum(np.asarray(a,dtype=np.int64)*(np.asarray(a,dtype=np.int64)-1)//2))
    truth_sizes = frame.groupby('reference').size()
    pred_sizes = frame.groupby('z').size()
    tp = comb(frame.groupby(['reference','z']).size())
    fp, fn = comb(pred_sizes)-tp, comb(truth_sizes)-tp
    visible = truth_sizes[truth_sizes >= 2]
    exact = 0
    for key, n in visible.items():
        labs = frame.loc[frame.reference == key, 'z'].unique()
        exact += int(len(labs) == 1 and pred_sizes.loc[labs[0]] == n)
    conflict, non_single, bad_gate, cross_edges = 0,0,0,0
    for label, group in d.assign(z=z).groupby('z'):
        if len(group) < 2:
            continue
        non_single += 1
        conflict += int(group.receiver.nunique() < len(group))
        i,j = pair_arrays(group)
        delta = np.abs(group.corrected_time_s.to_numpy()[i]-group.corrected_time_s.to_numpy()[j])
        bad_gate += int((delta > gate).sum())
        cross_edges += len(i)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn),exact_recovered=exact,visible_packets=len(visible),
                exact_recovery=exact/max(1,len(visible)),receiver_conflicts=conflict,
                non_singleton_clusters=non_single,receiver_violation=conflict/max(1,non_single),
                gate_violations=bad_gate,predicted_cross_edges=cross_edges,
                gate_violation=bad_gate/max(1,cross_edges))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days, inclusion = load()
    inclusion.to_csv(OUT/'inclusion.csv', index=False)
    gateways = sorted(set(pd.concat(days).gateway))
    offsets, offset_rows = fit_offsets(days, gateways)
    pd.DataFrame(offset_rows).to_csv(OUT/'training_clock_differences.csv', index=False)
    for d in days:
        d['receiver'] = d.gateway.map(dict(zip(gateways,range(len(gateways)))))
        d['corrected_time_s'] = d.time_s-d.gateway.map(offsets)
        d['window'] = np.floor(d.corrected_time_s/WINDOW).astype(int)
        d['observation_id'] = d.day+'/'+d.original_row.astype(str)
    cal_errors = []
    for d in days:
        if d['split'].iloc[0] == 'calibration':
            for _,g in d.groupby('reference_payload'):
                cal_errors.extend(abs(a.corrected_time_s-b.corrected_time_s) for a,b in combinations(g.itertuples(),2))
    gate = max(.001, float(np.quantile(cal_errors,.99)))
    windows = {d.day.iloc[0]: make_windows(d,gate) for d in days}
    train = [p for d in days if d['split'].iloc[0] == 'train' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
    calibration = [p for d in days if d['split'].iloc[0] == 'calibration' for _,_,p in windows[d.day.iloc[0]] if len(p.y)]
    models = {}
    fitted = {}
    for name, cols in [('timing',[0]),('metadata',[0,1,2])]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.,max_iter=500))
        model.fit(np.concatenate([p.X[:,cols] for p in train]),np.concatenate([p.y for p in train]))
        zz = np.concatenate([model.decision_function(p.X[:,cols])[p.gate(1.)] for p in calibration])
        yy = np.concatenate([p.y[p.gate(1.)] for p in calibration])
        res = minimize_scalar(lambda logT: float(np.mean(np.logaddexp(0,zz/np.exp(logT))-yy*zz/np.exp(logT))),
                              bounds=(np.log(.1),np.log(10)),method='bounded')
        assert res.success
        temp = float(np.exp(res.x))
        models[name] = (model,cols,temp)
        scaler, lr = model.steps[0][1], model.steps[1][1]
        fitted[name] = dict(columns=cols,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                            coefficient=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),temperature=temp)
    results = []
    window_rows = []
    normalized = []
    references = []
    for d in days:
        day, split = d.day.iloc[0], d['split'].iloc[0]
        normalized.append(d[['observation_id','day','split','receiver','time_s','corrected_time_s',
                              'frequency','spreading_factor','bandwidth','window']])
        references.append(d[['observation_id','reference_payload']])
        if split != 'test':
            continue
        for name,(model,cols,temp) in models.items():
            for decoder in ['components','assignment','constrained']:
                assignment = np.full(len(d),-1,int)
                next_label = 0
                gate_true, admitted = 0,0
                for group,scene,p in windows[day]:
                    if len(p.y):
                        probability = expit(model.decision_function(p.X[:,cols])/temp)
                        gate_true += int(np.sum(p.gate(1.)&p.y.astype(bool)))
                        admitted += int(p.gate(1.).sum())
                        # Reuse the same generic decoder; k=1 is the empirical LoED gate.
                        if decoder == 'constrained':
                            z = constrained_cluster(scene,p,probability,k=1.)
                        else:
                            pg = Pairs(p.X,p.Xraw,p.Xunweighted,p.y,p.i,p.j,p.time_z*3.5,p.freq_z,p.unseen)
                            z = connected_cluster(scene,pg,probability,assignment=(decoder == 'assignment'))
                    else:
                        z = np.arange(len(group))
                    # Match global indices by immutable observation IDs, not truth IDs.
                    locations = pd.Index(d.observation_id).get_indexer(group.observation_id)
                    assignment[locations] = z+next_label
                    next_label += int(z.max())+1
                assert (assignment >= 0).all()
                metric = counts(d,assignment,gate)
                if decoder == 'constrained':
                    assert metric['receiver_conflicts'] == metric['gate_violations'] == 0
                results.append(dict(day=day,scorer=name,decoder=decoder,gate_true_pairs=gate_true,
                                    admitted_pairs=admitted,**metric))
                normalized_day = pd.DataFrame(dict(observation_id=d.observation_id,predicted_cluster=assignment))
                normalized_day.to_csv(OUT/f'prediction_{day}_{name}_{decoder}.csv.gz',index=False)
        for group,_,p in windows[day]:
            window_rows.append(dict(day=day,window=int(group.window.iloc[0]),observations=len(group),
                                    cross_receiver_pairs=len(p.y),positive_pairs=int(p.y.sum())))
    pd.concat(normalized,ignore_index=True).to_csv(OUT/'observations.csv.gz',index=False)
    pd.concat(references,ignore_index=True).to_csv(OUT/'evaluation_reference.csv.gz',index=False)
    pd.DataFrame(window_rows).to_csv(OUT/'test_windows.csv',index=False)
    result = pd.DataFrame(results)
    result.to_csv(OUT/'per_day.csv',index=False)
    pooled = []
    for (name,decoder),g in result.groupby(['scorer','decoder']):
        sums = g[['tp','fp','fn','exact_recovered','visible_packets','receiver_conflicts','non_singleton_clusters',
                  'gate_violations','predicted_cross_edges','gate_true_pairs','admitted_pairs']].sum().to_dict()
        tp,fp,fn = sums['tp'],sums['fp'],sums['fn']
        pooled.append(dict(scorer=name,decoder=decoder,**sums,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                           f1=2*tp/max(1,2*tp+fp+fn),exact_recovery=sums['exact_recovered']/sums['visible_packets'],
                           receiver_violation=sums['receiver_conflicts']/max(1,sums['non_singleton_clusters']),
                           gate_violation=sums['gate_violations']/max(1,sums['predicted_cross_edges'])))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT/'summary.csv',index=False)
    (OUT/'fitted_parameters.json').write_text(json.dumps(dict(offsets_s=offsets,gate_s=gate,window_s=WINDOW,models=fitted),indent=2),encoding='utf-8')
    report = dict(status='PASS',dataset_doi='10.5281/zenodo.4121430',license='CC-BY-4.0',
                  archive_md5='7370baddc9b546354c26a71cd38981bc',
                  archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                  measured_data=True,synthetic_noise_or_class_probabilities_added=False,
                  reference='Full-PHY-payload identity under explicit uniqueness/span exclusions; not independent physical burst ground truth.',
                  splits=inclusion.groupby('split')[['raw_rows','eligible_reports','reference_packets','visible_reference_packets']].sum().to_dict(orient='index'),
                  raw_dates=inclusion.day.tolist(),time_gate_s=gate,window_s=WINDOW,
                  test_dates=sorted(result.day.unique()),test_observation_rows=int(inclusion[inclusion.split=='test'].eligible_reports.sum()),
                  test_reference_positives=int(inclusion[inclusion.split=='test'].reference_positive_pairs.sum()),
                  test_window_boundary_false_negatives_are_included=True,
                  identity_fields_excluded_from_features=True,scoring_features=['corrected time residual','frequency mismatch','decoded SF mismatch'],
                  limitation='Two held-out capture days; logical-packet reference; no waveform uncertainty/geometry or protocol-classifier validation.',
                  python=sys.version,platform=platform.platform(),pandas=pd.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(inclusion.to_string(index=False))
    print(pooled.to_string(index=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
