"""Bounded MILP audits at 20, 30, and 40 observations; no timeout is called exact."""
from itertools import combinations
import json,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from experiment import features,matrices,constrained_cluster
from review_common import ROOT,OUT,parameters,test_scenes,score,metric,objective,restart_greedy,slice_scene,environment

DEST=OUT/'exact'

def solve(scene,p,prob,seconds=5.):
    started=time.perf_counter();n=len(scene.truth)
    edges=list(combinations(range(n),2));index={e:k for k,e in enumerate(edges)}
    W,G=matrices(scene,p,prob)
    c=-np.array([W[a,b] for a,b in edges]);upper=np.array([G[a,b] for a,b in edges],float)
    rows=[];cols=[];vals=[];row=0
    for a,b,d in combinations(range(n),3):
        ids=[index[(a,b)],index[(a,d)],index[(b,d)]]
        for negative in range(3):
            for j,col in enumerate(ids):rows.append(row);cols.append(col);vals.append(-1. if j==negative else 1.)
            row+=1
    A=coo_matrix((vals,(rows,cols)),shape=(row,len(edges))).tocsc()
    solver_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        result=milp(c,integrality=np.ones(len(edges)),bounds=Bounds(np.zeros(len(edges)),upper),
                    constraints=LinearConstraint(A,-np.inf,np.ones(row)),
                    options={'time_limit':seconds,'mip_rel_gap':0.,'threads':1})
    solver_time=time.perf_counter()-solver_start
    labels=None
    if result.x is not None:
        assert np.max(np.abs(result.x-np.rint(result.x)))<1e-5
        parent=np.arange(n)
        def root(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        for x,(a,b) in zip(result.x,edges):
            if x>.5:parent[root(b)]=root(a)
        labels=np.array([root(i) for i in range(n)])
        check=metric(scene,p,labels)
        assert check['receiver_conflicts']==check['gate_violations']==0
        assert abs(objective(scene,p,prob,labels)+result.fun)<1e-5
    optimal=bool(result.status==0 and result.success)
    bound=float(-result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else np.nan
    value=float(-result.fun) if result.fun is not None else np.nan
    return labels,dict(optimal=optimal,status=int(result.status),message=result.message,
        incumbent_objective=value,objective_upper_bound=bound,solver_relative_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else np.nan,
        mip_nodes=int(result.mip_node_count) if getattr(result,'mip_node_count',None) is not None else -1,
        variables=len(edges),triangle_inequalities=row,mip_solver_s=solver_time,mip_total_s=time.perf_counter()-started,
        solver_option_warnings=[str(w.message) for w in notes])

def main():
    DEST.mkdir(parents=True,exist_ok=True);environment('exact');started=time.perf_counter()
    protocol=dict(replications=10,base_scenes_per_replication=2,observations=[20,30,40],
                  selection='First two existing test scenes with at least 40 reports; nested random report subsets selected without scores or labels.',
                  seed_base=85_000_000,solver_time_limit_s=5.,relative_gap_request=0.,highs_threads=1,
                  restart_first_edges=8,only_solver_proven_optima_called_exact=True)
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];details=[];saved={};case=0
    for rep in range(10):
        params=parameters(rep);selected=[]
        for idx,scene in enumerate(test_scenes(rep)):
            if len(scene.truth)>=40:selected.append((idx,scene))
            if len(selected)==2:break
        assert len(selected)==2
        for index,scene in selected:
            order=np.random.default_rng(85_000_000+rep*100+index).permutation(len(scene.truth))
            for n in [20,30,40]:
                sub=slice_scene(scene,order[:n]);p=features(sub);prob=score(p,params)
                t=time.perf_counter();greedy=constrained_cluster(sub,p,prob);greedy_s=time.perf_counter()-t
                jg=objective(sub,p,prob,greedy)
                t=time.perf_counter();restart,jr=restart_greedy(sub,p,prob,8);restart_s=time.perf_counter()-t
                assert jr>=jg-1e-8
                exact,record=solve(sub,p,prob)
                if record['optimal']:assert record['incumbent_objective']>=jr-1e-6
                row=dict(case=case,replicate=rep,source_scene=index,source_scene_seed=scene.seed,observations=n,
                         greedy_objective=jg,restart_objective=jr,greedy_s=greedy_s,restart_s=restart_s,**record)
                for method,z in [('greedy',greedy),('restart',restart),('milp',exact)]:
                    if z is not None:
                        m=metric(sub,p,z)
                        assert m['receiver_conflicts']==m['gate_violations']==0
                        row.update({method+'_'+key:value for key,value in m.items()})
                        saved[f'{case}_{method}']=z
                for name,value in [('greedy',jg),('restart',jr)]:
                    row[name+'_gap_to_optimum']=((record['incumbent_objective']-value)/max(abs(record['incumbent_objective']),1e-9)) if record['optimal'] else np.nan
                    row[name+'_gap_to_bound']=max(0.,(record['objective_upper_bound']-value)/max(abs(record['objective_upper_bound']),1e-9)) if np.isfinite(record['objective_upper_bound']) else np.nan
                details.append(dict(case=case,indices=order[:n].tolist(),scene_seed=scene.seed,probability=prob.tolist(),pair_i=p.i.tolist(),pair_j=p.j.tolist()))
                saved[f'{case}_truth']=sub.truth;saved[f'{case}_receiver']=sub.obs['receiver']
                rows.append(row);case+=1
        print(f'Medium MILP audit {rep+1}/10; {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'per_instance.csv',index=False)
    reports=[]
    for n,g in frame.groupby('observations'):
        proven=g[g.optimal]
        reports.append(dict(observations=int(n),instances=len(g),proven_optimal=len(proven),timeouts=int((g.status==1).sum()),
            greedy_gap_mean=float(proven.greedy_gap_to_optimum.mean()),greedy_gap_max=float(proven.greedy_gap_to_optimum.max()),
            restart_gap_mean=float(proven.restart_gap_to_optimum.mean()),restart_gap_max=float(proven.restart_gap_to_optimum.max()),
            greedy_f1=float(g.greedy_f1.mean()),restart_f1=float(g.restart_f1.mean()),milp_f1_proven=float(proven.milp_f1.mean()),
            greedy_ms=1000*float(g.greedy_s.median()),restart_ms=1000*float(g.restart_s.median()),milp_ms=1000*float(g.mip_total_s.median()),
            max_upper_bound_gap=float(g.greedy_gap_to_bound.max())))
    pd.DataFrame(reports).to_csv(DEST/'summary.csv',index=False)
    np.savez_compressed(DEST/'predictions.npz',**saved)
    (DEST/'instance_inputs.json').write_text(json.dumps(details),encoding='utf-8')
    (DEST/'execution.json').write_text(json.dumps(dict(status='PASS',seconds=time.perf_counter()-started,instances=len(frame),proven_optimal=int(frame.optimal.sum()),timeouts=int((frame.status==1).sum())),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
