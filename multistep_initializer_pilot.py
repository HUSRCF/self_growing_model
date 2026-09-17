"""Causal initializer fitted to fixed observed-route short-horizon targets."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def bounded_gauss_newton(initial,residual,iterations=6,bound=.02):
    v=initial.copy();trace=[]
    for _ in range(iterations):
        e=residual(v).reshape(len(v),-1);jac=[]
        for dim in range(2):
            step=np.zeros_like(v);step[:,dim]=1e-5
            jac.append((residual(v+step)-residual(v-step)).reshape(len(v),-1)/(2e-5))
        j=np.stack(jac,axis=-1)
        gram=np.einsum('nki,nkj->nij',j,j)/e.shape[1]+np.eye(2)[None]*1e-6
        rhs=np.einsum('nki,nk->ni',j,e)/e.shape[1]
        direction=np.linalg.solve(gram,rhs[...,None])[...,0]
        best=(e*e).mean(1);new=v.copy()
        for fraction in [.5,1.]:
            candidate=np.clip(v-fraction*direction,initial-bound,initial+bound)
            error=residual(candidate).reshape(len(v),-1);cost=(error*error).mean(1)
            take=cost<best;new[take]=candidate[take];best[take]=cost[take]
        assert np.isfinite(new).all() and np.isfinite(best).all()
        v=new;trace.append(float(best.mean()))
    return v,trace


def observed_routes(base,w):
    h=w['history'].copy();q=base.state_from_history(h)[0];sources=[];destinations=[]
    for y in w['truth'].transpose(1,0,2):
        nh=np.concatenate([h[:,1:],y[:,None]],axis=1);r=base.state_from_history(nh)[0]
        sources.append(q);destinations.append(r);h=nh;q=r
    return np.array(sources),np.array(destinations)


def fixed_route_residual(writer,w,sources,destinations,initial,steps):
    h=w['history'].copy();v=initial;out=[]
    for t in range(steps):
        y,v=writer.execute(h,sources[t],destinations[t],v)
        out.append(np.angle(np.exp(1j*(y-w['truth'][:,t]))))
        h=np.concatenate([h[:,1:],y[:,None]],axis=1)
    return np.stack(out,1)


class ResidualInitializer:
    def __init__(self,weak,coef,scale):
        self.weak=weak;self.coef=coef;self.scale=scale
        legacy_types()
        from v19.writers import history_features
        self.features=history_features
    def initialize(self,h):
        delta=self.features(self.weak.m,h)/self.scale@self.coef
        return self.weak.initialize(h)+np.clip(delta,-.02,.02)
    def execute(self,h,q,r,v):return self.weak.execute(h,q,r,v)


def main():
    root=Path('adaptive_search_results');path=root/'prefix_velocity_memory_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    s=AdaptiveBeam();_,Writer=legacy_types();from v19.writers import history_features,ridge
    block=load_block(path);weak=Writer(LegacyFeatureBridge(s.base),block,dict(learned_initialization=True))
    fit=prefix_windows(SPLITS['train'][:-3],steps=10,per_video=32)
    sources,destinations=observed_routes(s.base,fit);initial=weak.initialize(fit['history'])
    scale=block['initializer']['scale'];raw=history_features(weak.m,fit['history']);x=raw/scale
    models={'original':None,'base':weak};training={}
    for steps in [1,10]:
        cpu=time.process_time()
        residual=lambda v:fixed_route_residual(weak,fit,sources,destinations,v,steps)
        target,trace=bounded_gauss_newton(initial,residual)
        coef=ridge(x,target-initial,len(x)*.001)
        model=ResidualInitializer(weak,coef,scale);name=f'h{steps}';models[name]=model
        np.savez_compressed(root/f'multistep_initializer_{name}.npz',coef=coef,scale=scale)
        with np.load(root/f'multistep_initializer_{name}.npz') as z:
            np.testing.assert_array_equal(z['coef'],coef);np.testing.assert_array_equal(z['scale'],scale)
        training[name]=dict(trace=trace,initial_surrogate_mse=float(np.mean(residual(initial)**2)),
            optimized_surrogate_mse=float(np.mean(residual(target)**2)),
            regressed_surrogate_mse=float(np.mean(residual(model.initialize(fit['history']))**2)),
            target_rms_shift=float(np.sqrt(np.mean((target-initial)**2))),
            target_bound_hits=int((np.abs(target-initial)>=.02-1e-12).sum()),
            training_cpu_seconds=time.process_time()-cpu)
        print(name,training[name],flush=True)
    # Zero residual must leave the full baseline runtime unchanged.
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    zero=ResidualInitializer(weak,np.zeros_like(models['h1'].coef),scale)
    _,p,f,_=rollout_memory(s,hold,zero,131017)
    with np.load(root/'prefix_velocity_memory_memory_learned_131017.npz') as z:
        np.testing.assert_array_equal(p,z['prediction']);np.testing.assert_array_equal(f,z['failed'])
    runs={k:[] for k in models}
    for seed in range(201017,201021):
        for name,model in models.items():
            r,p,f,_=rollout_memory(s,hold,model,seed);runs[name].append(r)
            np.savez_compressed(root/f'multistep_initializer_{name}_{seed}.npz',prediction=p,failed=f,
                truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,r['objective'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(training=training,runs=runs,mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        fit_videos=SPLITS['train'][:-3],fit_windows=len(initial),fit_starts=fit['start'].tolist(),
        source_sha256=digest,source_unchanged=True,zero_residual_exact=True,
        note='Observed-q-route inverse targets, not on-policy/REINFORCE or unbiased closed-loop gradient. Same320TRAINprefix windows/features/ridge/bound for h1/h10, but h10 costs more. Future observed path only for training labels, inference causal. No noise/DEV/TEST/tuning/promotion.')
    (root/'multistep_initializer_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
