"""Fixed four-speed-bin initial covariance pilot, fitted only on TRAIN prefixes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from initial_velocity_distribution import InitialVelocityDistribution, residual_covariance
from audit_velocity_calibration import residual_stats
from velocity_memory_pilot import prefix_sequence, initializer_data, fit_initializer, load_block, rollout_memory
from velocity_memory_compat import LegacyFeatureBridge, legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def grouped_covariances(residual, weights, groups, global_cov, shrink=.1):
    out=[]
    for group in range(4):
        mask=groups==group
        if not mask.any():out.append(global_cov.copy());continue
        w=weights[mask]/weights[mask].sum();r=residual[mask];m=w@r
        c=((r-m)*w[:,None]).T@(r-m)
        out.append((1-shrink)*c+shrink*global_cov)
    return np.array(out)


class ConditionalVelocityDistribution:
    def __init__(self,writer,thresholds,covariances,seed):
        self.writer=writer;self.thresholds=np.asarray(thresholds);self.seed=seed
        if self.thresholds.shape!=(3,) or not np.all(np.diff(self.thresholds)>0):
            raise ValueError('three strictly increasing thresholds required')
        if np.shape(covariances)!=(4,2,2):raise ValueError('four covariances required')
        self.factors=np.array([InitialVelocityDistribution(writer,c,seed).factor for c in covariances])
    def initialize(self,h):
        v=self.writer.initialize(h)
        groups=np.searchsorted(self.thresholds,np.linalg.norm(h[:,-1]-h[:,-2],axis=1),side='right')
        noise=np.random.default_rng(self.seed).standard_normal(v.shape)
        out=v.copy()
        for group in range(4):
            mask=groups==group
            out[mask]+=noise[mask]@self.factors[group].T
        return out
    def execute(self,h,q,r,v):return self.writer.execute(h,q,r,v)


def main():
    root=Path('adaptive_search_results');model=root/'prefix_velocity_memory_model.npz'
    source=root/'initial_velocity_distribution.json'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,source]}
    global_cov=np.array(json.loads(source.read_text())['covariance'])
    s=AdaptiveBeam();base=LegacyFeatureBridge(s.base);_,Writer=legacy_types()
    sequences=[(v,prefix_sequence(load_video(v))) for v in SPLITS['train'][:-3]]
    residuals=[];speeds=[];weights=[]
    for video,y in sequences:
        init,_=fit_initializer(base,[(v,z) for v,z in sequences if v!=video])
        raw,target,last=initializer_data(base,[(video,y)])
        residuals.append(target-(last+raw/init['scale']@init['coef']))
        speeds.append(np.linalg.norm(last,axis=1))
        weights.append(np.full(len(raw),1/(len(sequences)*len(raw))))
    _,reconstructed=residual_covariance(residuals)
    np.testing.assert_array_equal(reconstructed,global_cov)
    speed=np.concatenate(speeds);thresholds=np.quantile(speed,[.25,.5,.75])
    groups=np.searchsorted(thresholds,speed,side='right')
    covs=grouped_covariances(np.concatenate(residuals),np.concatenate(weights),groups,global_cov)
    block=load_block(model);init=block['initializer']
    holdrows=[]
    for video in SPLITS['train'][-3:]:
        raw,target,last=initializer_data(base,[(video,prefix_sequence(load_video(video)))])
        holdrows.append((target-(last+raw/init['scale']@init['coef']),np.linalg.norm(last,axis=1),
                         np.full(len(raw),1/(3*len(raw)))))
    residual,speed,weight=[np.concatenate([r[i] for r in holdrows]) for i in range(3)]
    group=np.searchsorted(thresholds,speed,side='right');calibration={};nll={}
    for name,cs in [('global',np.repeat(global_cov[None],4,axis=0)),('conditional',covs)]:
        calibration[name]={};total=0.
        for g in range(4):
            mask=group==g;rr=residual[mask];ww=weight[mask];c=cs[g]
            calibration[name][str(g)]=residual_stats(rr,ww,c)
            d2=np.einsum('ni,ij,nj->n',rr,np.linalg.inv(c),rr)
            total+=float(ww@(.5*(d2+np.linalg.slogdet(c)[1]+2*np.log(2*np.pi))))
        nll[name]=total
    weak=Writer(base,block,dict(learned_initialization=True))
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    # Exact reduction at the actual initial histories, then full old trajectory replay.
    h=np.repeat(w['history'],8,axis=0)
    equal=ConditionalVelocityDistribution(weak,thresholds,np.repeat(global_cov[None],4,axis=0),1151017)
    np.testing.assert_array_equal(equal.initialize(h),InitialVelocityDistribution(weak,global_cov,1151017).initialize(h))
    _,p,f,_=rollout_memory(s,w,equal,151017)
    with np.load(root/'initial_velocity_distribution_distributed_velocity_151017.npz') as old:
        np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
    runs={name:[] for name in ['original','point','global','conditional']}
    for seed in range(171017,171021):
        writers=dict(original=None,point=weak,global_=InitialVelocityDistribution(weak,global_cov,seed+1000000),
                     conditional=ConditionalVelocityDistribution(weak,thresholds,covs,seed+1000000))
        writers['global']=writers.pop('global_')
        for name,writer in writers.items():
            r,p,f,_=rollout_memory(s,w,writer,seed);runs[name].append(r)
            np.savez_compressed(root/f'conditional_velocity_{name}_{seed}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
            print(seed,name,r['objective'],flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    comparisons={}
    for ref in ['original','point','global']:
        delta=np.array([a['objective']-b['objective'] for a,b in zip(runs['conditional'],runs[ref])])
        comparisons[ref]=dict(differences=delta.tolist(),mean=float(delta.mean()),conditional_rng_se=float(delta.std(ddof=1)/2))
    report=dict(thresholds=thresholds.tolist(),covariances=covs.tolist(),shrinkage_to_global=.1,
        calibration=calibration,hold_proxy_nll=nll,runs=runs,comparisons=comparisons,
        mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        source_hashes=hashes,sources_unchanged=True,global_covariance_reconstructed_exact=True,equal_covariance_replay_exact=True,
        note='TRAIN-only OOF residual calibration, fixed speed quartiles and 10% global shrinkage. No mean shift or scale tuning. One initial draw, frozen dynamics; local proxy NLL not trajectory objective. Four paired fresh RNG seeds on reused TRAIN hold, not independent videos. No DEV/TEST/promotion.')
    (root/'conditional_velocity_distribution.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
