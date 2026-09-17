"""Fixed learned acceleration, alternative integration within the selected edge."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_writer_local_bias import diagnostic_data
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def rk4_step(y,v,acc,dt=1.):
    k1y=v;k1v=acc(y,v)
    k2y=v+.5*dt*k1v;k2v=acc(y+.5*dt*k1y,k2y)
    k3y=v+.5*dt*k2v;k3v=acc(y+.5*dt*k2y,k3y)
    k4y=v+dt*k3v;k4v=acc(y+dt*k3y,k4y)
    return y+dt*(k1y+2*k2y+2*k3y+k4y)/6,v+dt*(k1v+2*k2v+2*k3v+k4v)/6


class RK4Writer:
    def __init__(self,weak):
        if weak.kind!='weak' or weak.config.get('reset_velocity',False) or weak.config.get('weak_blend',1.)!=1.:
            raise ValueError('unblended persistent weak writer required')
        self.weak=weak
        legacy_types()
        from v19.writers import library
        self.library=library
    def initialize(self,h):return self.weak.initialize(h)
    def execute(self,h,q,r,v):
        coef=self.weak.coef[None]+self.weak.config.get('local_fraction',1.)*self.weak.delta[q,r]
        def acc(y,velocity):return np.einsum('np,npd->nd',self.library(y,velocity,self.weak.d['order']),coef)
        return rk4_step(h[:,-1],v,acc)


def main():
    root=Path('adaptive_search_results');path=root/'prefix_velocity_memory_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    s=AdaptiveBeam();_,Writer=legacy_types()
    weak=Writer(LegacyFeatureBridge(s.base),load_block(path),dict(learned_initialization=True));rk=RK4Writer(weak)
    local={};old=json.loads((root/'writer_local_bias_audit.json').read_text())
    for split,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
        local[split]={}
        for video in videos:
            h,truth,_,_=diagnostic_data(load_video(video));sums=dict(original=0.,midpoint=0.,rk4=0.)
            for start in range(0,len(h),128):
                hh=h[start:start+128];target=truth[start:start+128];n=len(hh)
                q,m=s.machine.initialize(hh);pe,T,_=s.machine.read(hh,q,m);prob=np.einsum('ne,ner->nr',pe,T)
                history=np.repeat(hh,s.base.k,axis=0);qq=np.repeat(q,s.base.k);rr=np.tile(np.arange(s.base.k),n)
                v=weak.initialize(history)
                for name,p in [('original',s.base.execute_rule(history,qq,rr)),('midpoint',weak.execute(history,qq,rr,v)[0]),('rk4',rk.execute(history,qq,rr,v)[0])]:
                    error=np.angle(np.exp(1j*(p.reshape(n,s.base.k,2)-target[:,None])))
                    sums[name]+=float(np.sum(prob*np.mean(error**2,axis=2)))
            local[split][str(video)]={k:v/len(h) for k,v in sums.items()}
            for k,key in [('original','original'),('midpoint','learned')]:
                np.testing.assert_allclose(local[split][str(video)][k],old['per_video'][split][str(video)]['metrics'][key]['weighted_mse'],rtol=0,atol=1e-16)
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    _,p,f,_=rollout_memory(s,w,weak,131017)
    with np.load(root/'prefix_velocity_memory_memory_learned_131017.npz') as z:
        np.testing.assert_array_equal(p,z['prediction']);np.testing.assert_array_equal(f,z['failed'])
    runs={name:[] for name in ['original','midpoint','rk4']}
    writers=dict(original=None,midpoint=weak,rk4=rk)
    for i,seed in enumerate(range(181017,181021)):
        order=['original','midpoint','rk4'] if i%2==0 else ['rk4','midpoint','original']
        for name in order:
            cpu=time.process_time();result,p,f,_=rollout_memory(s,w,writers[name],seed)
            result['rollout_and_score_cpu_seconds']=time.process_time()-cpu;runs[name].append(result)
            np.savez_compressed(root/f'rk4_velocity_{name}_{seed}.npz',prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
            print(seed,name,result['objective'],result['rollout_and_score_cpu_seconds'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    comparisons={}
    for name in ['original','midpoint']:
        delta=np.array([a['objective']-b['objective'] for a,b in zip(runs['rk4'],runs[name])])
        comparisons[name]=dict(differences=delta.tolist(),mean=float(delta.mean()),conditional_rng_se=float(delta.std(ddof=1)/2))
    report=dict(local=local,runs=runs,comparisons=comparisons,model_sha256=digest,model_unchanged=True,
        midpoint_trajectory_exact=True,old_local_scores_replayed=True,
        mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        mean_cpu={k:float(np.mean([r['rollout_and_score_cpu_seconds'] for r in rs])) for k,rs in runs.items()},
        note='Fixed acceleration/qedge/learned initialization,dt1sample; four RK4 stages versus two midpoint, no new event reads inside step. Point initialization only. TRAIN prefixes/hold, no proxy inputs to writer, no fit/noise/tuning/DEV/TEST/promotion. CPU includes scoring, excludes compression; descriptive timing, not isolated integrator benchmark.')
    (root/'rk4_velocity_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
