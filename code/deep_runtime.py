"""Fresh fixed-registry latency measurements; descriptor generation is excluded."""
from pathlib import Path
from dataclasses import replace
from itertools import combinations
from math import comb
import json,time,platform,sys,warnings
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from experiment import Config,simulate,features,constrained_cluster,matrices
from review_common import parameters,score,metric,objective,restart_greedy,slice_scene

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/deep_revision/runtime'

def export_table():
    summary=pd.read_csv(DEST/'summary.csv')
    names={'greedy':'Greedy','restart':'Restart','milp':'MILP'}
    lines=[r'\begin{tabularx}{\linewidth}{@{}rYrrrrr@{}}\toprule',
        r'$N$ & Decoder & Median (ms) & Maximum (ms) & Event F1 & Proven / 10 & Timeouts \\\midrule']
    for n in [40,80,120]:
        for name in ['greedy','restart','milp']:
            row=summary[(summary.observations==n)&(summary.method==name)].iloc[0]
            proven=str(int(row.proven_optima)) if name=='milp' else '--'
            timeouts=str(int(row.timeouts)) if name=='milp' else '--'
            lines.append(f'{n} & {names[name]} & {row.report_path_median_ms:.2f} & {row.report_path_max_ms:.2f} & {row.f1_mean:.4f} & {proven} & {timeouts} '+r'\\')
    lines.append(r'\bottomrule\end{tabularx}')
    (ROOT/'tables/deep_runtime.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def memory_probe():
    """One Windows process-memory check; does not alter reported latency records."""
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=wintypes.HANDLE
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];get.restype=wintypes.BOOL
    def read():
        info=Counters();info.cb=ctypes.sizeof(info)
        if not get(current(),ctypes.byref(info),info.cb):raise ctypes.WinError()
        return {name:int(getattr(info,name)) for name in ['WorkingSetSize','PeakWorkingSetSize','PagefileUsage','PeakPagefileUsage']}
    initial=read();n=120;seed=120_000_000+100*n;cfg=replace(Config(),events=int(np.ceil(n/3.2))+5)
    for attempt in range(100):
        s=simulate(seed+attempt,cfg)
        if len(s.truth)>=n:break
    order=np.random.default_rng(seed+999).permutation(len(s.truth))[:n];s=slice_scene(s,order)
    p=features(s);pr=score(p,parameters(0));z,detail=timed_milp(s,p,pr,seconds=1.)
    record=dict(status='PASS',observations=n,scene_seed=s.seed,before_bytes=initial,after_bytes=read(),
        measured_scope='One isolated Windows process; includes loaded Python and numerical libraries.',
        not_a_universal_memory_bound=True,proven_optimal=detail['optimal'],
        variables=detail['variables'],triangle_inequalities=detail['triangle_inequalities'],
        sparse_initial_arrays_bytes=detail['triangle_inequalities']*3*(4+4+8))
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'memory_probe.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

def timed_milp(scene,p,prob,seconds=1.):
    """Time construction, solution and label decoding; verify truth metrics later.

    The full three-inequality-per-triple formulation matches review_exact.solve.
    Preallocated sparse arrays avoid Python-list storage for millions of entries.
    """
    started=time.perf_counter();n=len(scene.truth)
    edges=list(combinations(range(n),2));W,G=matrices(scene,p,prob)
    cost=-np.asarray([W[a,b] for a,b in edges]);upper=np.asarray([G[a,b] for a,b in edges],float)
    nrows=3*comb(n,3);nnz=3*nrows
    rr=np.repeat(np.arange(nrows,dtype=np.int32),3)
    cc=np.empty(nnz,dtype=np.int32);vv=np.ones(nnz,dtype=float)
    offset=0
    def edge_index(a,b):return a*(2*n-a-1)//2+b-a-1
    for a,b,c in combinations(range(n),3):
        ids=(edge_index(a,b),edge_index(a,c),edge_index(b,c))
        cc[offset:offset+9]=ids*3
        vv[offset]=-1.;vv[offset+4]=-1.;vv[offset+8]=-1.
        offset+=9
    assert offset==nnz
    matrix=coo_matrix((vv,(rr,cc)),shape=(nrows,len(edges))).tocsc()
    del rr,cc,vv
    solver_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        result=milp(cost,integrality=np.ones(len(edges)),bounds=Bounds(np.zeros(len(edges)),upper),
            constraints=LinearConstraint(matrix,-np.inf,np.ones(nrows)),
            options={'time_limit':seconds,'mip_rel_gap':0.,'threads':1})
    solver_s=time.perf_counter()-solver_start
    labels=None
    if result.x is not None:
        parent=np.arange(n)
        def root(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        for value,(a,b) in zip(result.x,edges):
            if value>.5:parent[root(b)]=root(a)
        labels=np.asarray([root(i) for i in range(n)])
    total_s=time.perf_counter()-started
    # Evaluation and correctness checks are outside the reported runtime.
    if labels is not None:
        assert np.max(np.abs(result.x-np.rint(result.x)))<1e-5
        check=metric(scene,p,labels)
        assert check['receiver_conflicts']==check['gate_violations']==0
        assert abs(objective(scene,p,prob,labels)+result.fun)<1e-5
    optimal=bool(result.status==0 and result.success)
    return labels,dict(optimal=optimal,status=int(result.status),message=result.message,
        incumbent_objective=float(-result.fun) if result.fun is not None else np.nan,
        objective_upper_bound=float(-result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else np.nan,
        solver_relative_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else np.nan,
        mip_nodes=int(result.mip_node_count) if getattr(result,'mip_node_count',None) is not None else -1,
        variables=len(edges),triangle_inequalities=nrows,mip_solver_s=solver_s,mip_total_s=total_s,
        solver_option_warnings=[str(w.message) for w in notes])

def run():
    DEST.mkdir(parents=True,exist_ok=True)
    protocol=dict(replications=10,report_counts=[40,80,120],seed_base=120_000_000,
        selection='Report count alone; first sufficiently large scene, then a random report subset.',
        scorer='frozen original entropy-weighted linear',solver_limit_s=1.,
        numerical_threads=1,restart_first_edges=8,greedy_and_restart_path_repeats=3,milp_path_repeats=1,
        timing_scope='Native report features, frozen pair scoring, and event decoding with a fixed class registry. Generation, file I/O, waveform detection, and offline map adaptation are excluded.',
        evaluation_and_feasibility_audit_excluded_from_timings=True,
        milp_storage='Preallocated int32 sparse indices and float64 values; same full triangle formulation as original audit.',
        no_runtime_target_assumed=True,python=sys.version,platform=platform.platform())
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];saved={};started=time.perf_counter()
    # Initialize the numerical decoder outside all reported timing intervals.
    s=simulate(129_000_000,Config());p=features(s);pr=score(p,parameters(0))
    constrained_cluster(s,p,pr)
    warm=slice_scene(s,np.arange(min(8,len(s.truth))));wp=features(warm)
    timed_milp(warm,wp,score(wp,parameters(0)),seconds=1.)
    for rep in range(10):
        par=parameters(rep)
        for n in protocol['report_counts']:
            seed=120_000_000+100_000*rep+100*n
            cfg=replace(Config(),events=int(np.ceil(n/3.2))+5)
            for attempt in range(100):
                s=simulate(seed+attempt,cfg)
                if len(s.truth)>=n:break
            else:raise RuntimeError('No sufficiently large scene')
            order=np.random.default_rng(seed+999).permutation(len(s.truth))[:n]
            s=slice_scene(s,order);base=dict(replicate=rep,observations=n,scene_seed=s.seed,
                source_event_count=cfg.events,selected_indices=order.tolist())
            common_g=[];common_r=[];gt=[];rt=[];gpath=[];rpath=[]
            for repeat in range(3):
                start_path=time.perf_counter();p=features(s);pr=score(p,par);common_g.append(time.perf_counter()-start_path)
                t=time.perf_counter();zg=constrained_cluster(s,p,pr);gt.append(time.perf_counter()-t);gpath.append(time.perf_counter()-start_path)
                start_path=time.perf_counter();p=features(s);pr=score(p,par);common_r.append(time.perf_counter()-start_path)
                t=time.perf_counter();zr,jr=restart_greedy(s,p,pr,8);rt.append(time.perf_counter()-t);rpath.append(time.perf_counter()-start_path)
            start_path=time.perf_counter();p=features(s);pr=score(p,par);common_m=time.perf_counter()-start_path
            zm,detail=timed_milp(s,p,pr,seconds=1.)
            jg=objective(s,p,pr,zg)
            for method,z,seconds,prep,path_s in [('greedy',zg,float(np.median(gt)),float(np.median(common_g)),float(np.median(gpath))),
                    ('restart',zr,float(np.median(rt)),float(np.median(common_r)),float(np.median(rpath))),
                    ('milp',zm,detail['mip_total_s'],common_m,common_m+detail['mip_total_s'])]:
                row={k:v for k,v in base.items() if k!='selected_indices'}
                row.update(method=method,features_and_scoring_ms=1000*prep,
                    decoding_ms=1000*seconds,report_path_ms=1000*path_s,
                    proven_optimal=bool(detail['optimal']) if method=='milp' else False,
                    solver_status=detail['status'] if method=='milp' else -1,
                    solver_ms=1000*detail['mip_solver_s'] if method=='milp' else np.nan,
                    has_feasible_output=z is not None,
                    objective_upper_bound=detail['objective_upper_bound'],
                    solver_relative_gap=detail['solver_relative_gap'] if method=='milp' else np.nan,
                    solver_message=detail['message'] if method=='milp' else '',
                    solver_nodes=detail['mip_nodes'] if method=='milp' else -1,
                    variables=detail['variables'] if method=='milp' else 0,
                    triangle_inequalities=detail['triangle_inequalities'] if method=='milp' else 0)
                if z is not None:
                    val=objective(s,p,pr,z);row['objective']=val
                    row.update(metric(s,p,z))
                    assert row['receiver_conflicts']==row['gate_violations']==0
                    row['gap_to_optimum']=(detail['incumbent_objective']-val)/max(abs(detail['incumbent_objective']),1e-9) if detail['optimal'] else np.nan
                    saved[f'{rep}_{n}_{method}']=z
                rows.append(row)
            saved[f'{rep}_{n}_truth']=s.truth
            saved[f'{rep}_{n}_receiver']=s.obs['receiver']
            saved[f'{rep}_{n}_source_indices']=order
            if detail['optimal']:assert detail['incumbent_objective']>=max(jg,jr)-1e-5
            pd.DataFrame(rows).to_csv(DEST/'per_instance.csv',index=False)
        print(f'Latency {rep+1}/10; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for (n,method),g in frame.groupby(['observations','method']):
        summary.append(dict(observations=int(n),method=method,instances=len(g),
            feature_score_median_ms=float(g.features_and_scoring_ms.median()),
            decoder_median_ms=float(g.decoding_ms.median()),
            report_path_median_ms=float(g.report_path_ms.median()),
            report_path_max_ms=float(g.report_path_ms.max()),
            report_path_p95_ms=float(g.report_path_ms.quantile(.95)),
            timeouts=int((g.solver_status==1).sum()),proven_optima=int(g.proven_optimal.sum()),
            feasible_outputs=int(g.has_feasible_output.sum()),f1_mean=float(g.f1.mean()),
            gap_proven_mean=float(g.gap_to_optimum.mean())))
    pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
    np.savez_compressed(DEST/'predictions.npz',**saved)
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',instances=30,
        decoder_rows=len(frame),elapsed_s=time.perf_counter()-started),indent=2),encoding='utf-8')
    export_table()

if __name__=='__main__':
    with threadpool_limits(limits=1):
        if '--memory-probe' in sys.argv:memory_probe()
        elif '--table-only' in sys.argv:export_table()
        else:run()
