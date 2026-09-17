"""Fit causal initial state to the frozen writer's expected one-step displacement."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_pilot import prefix_sequence,load_block,save_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from audit_writer_local_bias import diagnostic_data
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def solve_velocity(displacement,initial,acc,iterations=8,bound=.02):
    v=initial.copy()
    for _ in range(iterations):v=np.clip(displacement-.5*acc(v),initial-bound,initial+bound)
    residual=v+.5*acc(v)-displacement
    return v,residual


def main():
    root=Path('adaptive_search_results');source=root/'prefix_velocity_memory_model.npz'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    s=AdaptiveBeam();base=LegacyFeatureBridge(s.base);_,Writer=legacy_types()
    from v19.writers import history_features,library,ridge
    block=load_block(source);weak=Writer(base,block,dict(learned_initialization=True))
    raws=[];targets=[];lasts=[];solvers=[]
    for video in SPLITS['train'][:-3]:
        y=prefix_sequence(load_video(video));t=np.arange(32,len(y)-4)
        for start in range(0,len(t),128):
            ix=t[start:start+128];h=np.stack([y[i-31:i+1] for i in ix])
            q,m=s.machine.initialize(h);pe,T,_=s.machine.read(h,q,m);prob=np.einsum('ne,ner->nr',pe,T)
            coef=weak.coef[None]+np.einsum('nr,nrpd->npd',prob,weak.delta[q])
            def acc(v):return np.einsum('np,npd->nd',library(h[:,-1],v,block['order']),coef)
            initial=weak.initialize(h);delta=y[ix+1]-y[ix]
            target,residual=solve_velocity(delta,initial,acc)
            assert np.isfinite(target).all() and np.isfinite(residual).all()
            solvers.append(dict(n=len(h),residual_square=float(np.sum(residual**2)),
                                max_abs_residual=float(np.max(np.abs(residual))),
                                bound_hits=int((np.abs(target-initial)>=.02-1e-12).sum())))
            raws.append(history_features(base,h));targets.append(target);lasts.append(y[ix]-y[ix-1])
    raw=np.concatenate(raws);target=np.concatenate(targets);last=np.concatenate(lasts)
    scale=np.maximum(np.sqrt((raw**2).mean(0)),1e-7)
    np.testing.assert_array_equal(scale,block['initializer']['scale'])
    coef=ridge(raw/scale,target-last,len(raw)*.001)
    newblock=dict(block,initializer=dict(scale=scale,coef=coef));path=root/'writer_compatible_initializer_model.npz'
    save_block(path,newblock);adapted=Writer(base,load_block(path),dict(learned_initialization=True))
    np.testing.assert_array_equal(adapted.d['initializer']['coef'],coef)
    for k in ['coef','pair_delta','counts']:np.testing.assert_array_equal(block[k],adapted.d[k])
    local={}
    for video in SPLITS['train'][-3:]:
        h,truth,_,_=diagnostic_data(load_video(video));sums=dict(proxy=0.,compatible=0.)
        for start in range(0,len(h),128):
            hh=h[start:start+128];yy=truth[start:start+128];n=len(hh)
            q,m=s.machine.initialize(hh);pe,T,_=s.machine.read(hh,q,m);prob=np.einsum('ne,ner->nr',pe,T)
            history=np.repeat(hh,s.base.k,axis=0);qq=np.repeat(q,s.base.k);rr=np.tile(np.arange(s.base.k),n)
            for name,writer in [('proxy',weak),('compatible',adapted)]:
                p,_=writer.execute(history,qq,rr,writer.initialize(history))
                err=np.angle(np.exp(1j*(p.reshape(n,s.base.k,2)-yy[:,None])))
                sums[name]+=float(np.sum(prob*np.mean(err**2,axis=2)))
        local[str(video)]={k:v/len(h) for k,v in sums.items()}
    old=json.loads((root/'writer_local_bias_audit.json').read_text())
    for v,r in local.items():np.testing.assert_allclose(r['proxy'],old['per_video']['hold'][v]['metrics']['learned']['weighted_mse'],rtol=0,atol=1e-16)
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs={k:[] for k in ['original','proxy','compatible']}
    for seed in range(191017,191021):
        for name,writer in [('original',None),('proxy',weak),('compatible',adapted)]:
            r,p,f,_=rollout_memory(s,w,writer,seed);runs[name].append(r)
            np.savez_compressed(root/f'compatible_initializer_{name}_{seed}.npz',prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
            print(seed,name,r['objective'],flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    comparisons={}
    for name in ['original','proxy']:
        d=np.array([a['objective']-b['objective'] for a,b in zip(runs['compatible'],runs[name])])
        comparisons[name]=dict(differences=d.tolist(),mean=float(d.mean()),conditional_rng_se=float(d.std(ddof=1)/2))
    report=dict(fit_rows=len(raw),solver=dict(iterations=8,bound=.02,
        rmse=float(np.sqrt(sum(r['residual_square'] for r in solvers)/(2*len(raw)))),
        max_abs_residual=max(r['max_abs_residual'] for r in solvers),bound_hits=sum(r['bound_hits'] for r in solvers)),
        fit_target_rmse=float(np.sqrt(np.mean((last+raw/scale@coef-target)**2))),local_hold=local,runs=runs,comparisons=comparisons,
        mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        source_sha256=digest,source_unchanged=True,new_model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='TRAIN10prefix inverse expected-displacement labels only, original137causal features/scale/ridge N*.001; coefficients of acceleration unchanged. Eight clipped fixed-point steps, no tuning. Mean displacement matching not full expected squared-error optimum. New v is effective state, not claimed physical velocity. No future inference, no noise/DEV/TEST/promotion.')
    (root/'writer_compatible_initializer.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
