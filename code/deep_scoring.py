"""Frozen evidence adapters for the targeted descriptor experiments.

Linear fits are read unchanged. The original boosted classifier was not serialized.
It is reconstructed from the original training scenes and hyperparameters. Its
recorded temperature remains fixed. New adaptation or test labels never fit it.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from experiment import Config,Scene,Pairs,Scorer,simulate,features,constrained_cluster
from partial_mapping_extension import common

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'results/deep_revision/models'
EPS=1e-12

@dataclass
class FrozenScorer:
    kind:str
    params:dict
    model:object=None

    def logits_from(self,physical,compatibility,concentration):
        sem=compatibility if self.kind=='unweighted' else concentration*compatibility
        X=np.column_stack([physical,sem])
        if self.kind=='boosted':
            if not len(X):return np.empty(0)
            return logit(np.clip(self.model.predict_proba(X)[:,1],EPS,1-EPS))
        q=self.params
        X=X[:,q['columns']]
        return ((X-np.asarray(q['mean']))/np.asarray(q['scale']))@np.asarray(q['coefficient'])+q['intercept']

    def probabilities_from(self,physical,compatibility,concentration):
        return expit(self.logits_from(physical,compatibility,concentration)/self.params['temperature'])

    def weights_from(self,physical,compatibility,concentration):
        return logit(np.clip(self.probabilities_from(physical,compatibility,concentration),EPS,1-EPS))

    def probabilities(self,batch,maps):
        pi=common(batch,maps);K=pi.shape[1]
        r=1+np.sum(pi*np.log(np.maximum(pi,EPS)),axis=1)/np.log(K)
        comp=np.log(np.maximum(K*np.sum(pi[batch.i]*pi[batch.j],axis=1),EPS))
        return self.probabilities_from(batch.physical,comp,r[batch.i]*r[batch.j])

    def weights(self,batch,maps):
        return logit(np.clip(self.probabilities(batch,maps),EPS,1-EPS))

    def decode(self,batch,maps):
        n=len(batch.receiver);zeros=np.zeros(len(batch.i))
        dummy=Scene({'receiver':batch.receiver},np.zeros(n,int),np.zeros(n,bool),np.empty((0,2)),0)
        p=Pairs(batch.physical,batch.physical,batch.physical,zeros,batch.i,batch.j,
                batch.time_z,batch.freq_z,np.zeros(len(zeros),bool))
        return constrained_cluster(dummy,p,self.probabilities(batch,maps),k=batch.gate_k)

def load_scorers(rep,include_boosted=True):
    params=json.loads((ROOT/f'results/linear_models_r{rep:02d}.json').read_text())
    result={k:FrozenScorer(k,params[k]) for k in ['physics','joint','unweighted']}
    if not include_boosted:return result
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    path=MODEL_DIR/f'boosted_r{rep:02d}.pkl'
    meta_path=MODEL_DIR/f'boosted_r{rep:02d}.json'
    if path.exists() and meta_path.exists():
        with path.open('rb') as f:model=pickle.load(f)
        meta=json.loads(meta_path.read_text())
    else:
        start=time.perf_counter();base=1_000_000+100_000*rep
        train=[features(simulate(base+j,Config())) for j in range(120)]
        fitted=Scorer('boosted');fitted.fit(train)
        temperatures=pd.read_csv(ROOT/'results/temperatures.csv')
        T=float(temperatures[(temperatures.replicate==rep)&(temperatures.method=='boosted')].temperature.iloc[0])
        fitted.temperature=T;model=fitted.model
        meta=dict(replicate=rep,temperature=T,training_seed_base=base,training_scenes=120,
                  temperature_source='unchanged original temperatures.csv',
                  model_hyperparameters=model.get_params(),training_seconds=time.perf_counter()-start,
                  generator_sha256=hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
                  new_adaptation_or_test_used_for_fitting=False)
        if rep==0:
            test=[features(simulate(base+20_000+j,Config())) for j in range(80)]
            reproduced=np.concatenate([fitted.score(p,True,True) for p in test])
            old=np.load(ROOT/'results/prediction_boosted_calibrated_r00.npz')['prob']
            delta=float(np.max(np.abs(old-reproduced)))
            meta['r00_original_80_scene_probability_max_difference']=delta
            if delta>1e-8:raise AssertionError(f'Boosted reconstruction differs by {delta}')
        # Local generated models only; do not load external pickle files.
        with path.open('wb') as f:pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
        meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    result['boosted']=FrozenScorer('boosted',{'temperature':meta['temperature']},model)
    return result

def verify_frozen_models():
    """Compare every new adapter with the retained full nominal factorial record."""
    from partial_mapping_extension import controlled_adapter
    rows=[];identity=np.tile(np.arange(4),(4,1))
    for rep in range(10):
        scorers=load_scorers(rep);saved=np.load(ROOT/f'results/factorial/predictions_r{rep:02d}.npz')
        max_difference={k:0. for k in scorers};partition_equal={k:0 for k in scorers}
        for j in range(80):
            scene=simulate(1_000_000+100_000*rep+20_000+j,Config())
            batch,p=controlled_adapter(scene,identity)
            a,b=map(int,saved['pair_boundaries'][j:j+2]);c,d=map(int,saved['observation_boundaries'][j:j+2])
            assert b-a==len(p.i) and d-c==len(scene.truth)
            for kind,scorer in scorers.items():
                new=scorer.probabilities(batch,identity);old=saved[f'score_{kind}'][a:b]
                delta=float(np.max(np.abs(new-old)));max_difference[kind]=max(max_difference[kind],delta)
                new_labels=scorer.decode(batch,identity);old_labels=saved[f'{kind}_constrained'][c:d]
                equal=np.array_equal(new_labels[:,None]==new_labels[None,:],old_labels[:,None]==old_labels[None,:])
                partition_equal[kind]+=int(equal)
                assert delta<1e-8 and equal,(rep,j,kind,delta)
        rows.extend(dict(replicate=rep,scorer=k,maximum_pair_probability_difference=max_difference[k],equal_partitions=partition_equal[k],scenes=80) for k in scorers)
    record=dict(status='PASS',comparisons=3200,details=rows,
                maximum_probability_difference=max(x['maximum_pair_probability_difference'] for x in rows),
                original_scenes_used_for_verification_only=True)
    (MODEL_DIR/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='details'},indent=2))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):verify_frozen_models()
