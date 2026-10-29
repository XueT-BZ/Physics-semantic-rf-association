#!/usr/bin/env python3
"""Verify that every magnified plotted gap is derived from saved observations.

None of the lower difference panels adds or subtracts a visual vertical offset to
an experimental series. Horizontal marker nudges affect only graphical rendering.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';D=R/'figure_data'
s=pd.read_csv(R/'stress_summary.csv')
rel=pd.read_csv(D/'reliability.csv')
rd=pd.read_csv(D/'reliability_difference.csv')
pre=rel[rel.method=='joint_gated'].set_index('bin')
post=rel[rel.method=='joint_calibrated'].set_index('bin')
expected=(post.loc[rd.bin,'positive_fraction'].to_numpy()-pre.loc[rd.bin,'positive_fraction'].to_numpy())
np.testing.assert_allclose(rd.delta_after_minus_before.to_numpy(),expected,rtol=0,atol=1e-13)

sem=pd.read_csv(D/'semantic_difference.csv')
p=s[(s.group=='semantic')&(s.method=='physics')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
j=s[(s.group=='semantic')&(s.method=='joint')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
np.testing.assert_allclose(sem.error_probability,p.x,atol=1e-13)
np.testing.assert_allclose(sem.joint_minus_physical_f1,j.f1.to_numpy()-p.f1.to_numpy(),atol=1e-13)

cd=pd.read_csv(D/'clock_difference.csv')
for metric in ['f1','gate_recall']:
 d=cd[cd.metric==metric].sort_values('clock_s.d._microseconds')
 f=s[(s.group=='clock_underreported')&(s.method=='joint')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
 m=s[(s.group=='clock_matched')&(s.method=='joint')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
 np.testing.assert_allclose(d['clock_s.d._microseconds'],m.x,atol=1e-13)
 np.testing.assert_allclose(d.matched_minus_fixed,m[metric].to_numpy()-f[metric].to_numpy(),atol=1e-13)

runtime=pd.read_csv(D/'runtime_difference.csv')
p=s[(s.group=='scaling')&(s.method=='physics')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
j=s[(s.group=='scaling')&(s.method=='joint')].assign(x=lambda df:df.value.astype(float)).sort_values('x')
np.testing.assert_allclose(runtime.physical_ms,p.runtime_ms,atol=1e-13)
np.testing.assert_allclose(runtime.joint_ms,j.runtime_ms,atol=1e-13)
np.testing.assert_allclose(runtime.joint_minus_physical_percent,100*(j.runtime_ms.to_numpy()/p.runtime_ms.to_numpy()-1),atol=1e-13)
report={
 'status':'PASS',
 'calibration_difference_bins':len(rd),
 'semantic_difference_points':len(sem),
 'clock_difference_points':len(cd),
 'runtime_relative_difference_points':len(runtime),
 'recorded_runtime_difference_percent':dict(zip(runtime.mean_observations.astype(str),np.round(runtime.joint_minus_physical_percent,4))),
 'presentation':'Original data y values and curve x locations unchanged. Only errorbar/marker display locations are nudged, as disclosed in manuscript captions.'
}
target=ROOT/'audit'/'curve_integrity.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
