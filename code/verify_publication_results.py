"""Independent audit of saved public predictions, reference isolation, and tables."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from loed_benchmark import make_windows, WINDOW

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/loed'


def main():
    archive=ROOT/'data/public/loed/LoED_LoRaWAN_at_edge_dataset-SAMPLE.zip'
    assert hashlib.md5(archive.read_bytes()).hexdigest()=='7370baddc9b546354c26a71cd38981bc'
    o=pd.read_csv(OUT/'observations.csv.gz')
    reference=pd.read_csv(OUT/'evaluation_reference.csv.gz')
    assert o.observation_id.is_unique and reference.observation_id.is_unique
    assert set(o.observation_id)==set(reference.observation_id)
    assert not {'reference_payload','physical_payload','fcnt','device_address','fport'}.intersection(o.columns)
    d=o.merge(reference,on='observation_id',validate='one_to_one')
    seen={s:set(g.reference_payload) for s,g in d.groupby('split')}
    assert not seen['train']&seen['calibration'] and not seen['train']&seen['test'] and not seen['calibration']&seen['test']
    assert d[d.split=='test'].shape[0]==23677
    inclusion=pd.read_csv(OUT/'inclusion.csv')
    assert inclusion.raw_rows.sum()==326870
    for row in inclusion.itertuples():
        assert row.valid_unconfirmed_uplink_rows-row.exact_duplicate_rows_removed-row.invalid_observable_rows_removed-row.ambiguous_rows_removed-row.cross_split_shared_payload_rows_removed==row.eligible_reports
    params=json.loads((OUT/'fitted_parameters.json').read_text())
    gate=params['gate_s']
    listed=pd.read_csv(OUT/'per_day.csv')
    checked=0
    for path in sorted(OUT.glob('prediction_*.csv.gz')):
        parts=path.name.removeprefix('prediction_').removesuffix('.csv.gz').split('_')
        day,scorer,decoder=parts
        p=pd.read_csv(path)
        day_frame=d[d.day==day].merge(p,on='observation_id',validate='one_to_one')
        assert len(day_frame)==len(p)==len(d[d.day==day])
        truth_sizes=day_frame.groupby('reference_payload').size().to_numpy()
        predicted_sizes=day_frame.groupby('predicted_cluster').size().to_numpy()
        shared=day_frame.groupby(['reference_payload','predicted_cluster']).size().to_numpy()
        choose2=lambda sizes:int((sizes*(sizes-1)//2).sum())
        tp=choose2(shared);fp=choose2(predicted_sizes)-tp;fn=choose2(truth_sizes)-tp
        row=listed[(listed.day==day)&(listed.scorer==scorer)&(listed.decoder==decoder)].iloc[0]
        assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
        assert np.isclose(2*tp/(2*tp+fp+fn),row.f1)
        conflict=0;bad=0
        for _,g in day_frame.groupby('predicted_cluster'):
            if len(g)>1:
                conflict+=int(g.receiver.nunique()<len(g))
                ii,jj=np.triu_indices(len(g),1)
                keep=g.receiver.to_numpy()[ii]!=g.receiver.to_numpy()[jj]
                bad+=int((np.abs(g.corrected_time_s.to_numpy()[ii[keep]]-g.corrected_time_s.to_numpy()[jj[keep]])>gate).sum())
        assert conflict==row.receiver_conflicts and bad==row.gate_violations
        if decoder=='constrained':assert conflict==bad==0
        checked+=1
    assert checked==12
    # Mutating reference memberships cannot change public feature or gate arrays.
    group=d[d.day=='2020-08-03'].groupby('window').get_group(1000).reset_index(drop=True)
    altered=group.copy();altered['reference_payload']=['changed_'+str(i) for i in range(len(group))]
    a=make_windows(group,gate)[0][2];b=make_windows(altered,gate)[0][2]
    assert np.array_equal(a.X,b.X) and np.array_equal(a.gate(1.),b.gate(1.))
    # The measured table is generated from full precision outputs.
    table=(ROOT/'tables/loed_results.tex').read_text()
    pooled=pd.read_csv(OUT/'summary.csv')
    for row in pooled.itertuples():assert f'{row.f1:.4f}' in table
    factorial=pd.read_csv(ROOT/'results/factorial/per_scene.csv')
    assert len(factorial)==9600
    assert len(factorial.groupby(['replicate','scene']))==800
    assert all(len(g)==12 for _,g in factorial.groupby(['replicate','scene']))
    assert factorial[factorial.decoder=='constrained'][['receiver_violation','gate_violation']].to_numpy().max()==0
    report=dict(status='PASS',raw_public_rows_accounted=326870,public_prediction_files_verified=checked,
                held_out_public_reports=23677,independent_tp_fp_fn_recomputed=True,
                split_identity_disjointness=True,reference_membership_feature_invariance=True,
                cross_window_false_negatives_included=True,factorial_scene_conditions_verified=9600,
                official_mdpi_class_present=(ROOT/'Definitions/mdpi.cls').exists(),
                scope='Independent saved-output and identity-isolation audit; not independent physical-burst ground truth.')
    target=ROOT/'audit/publication_revision_20261020/results_verification.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
