"""TRAIN-prefix refit of weak acceleration writer with causal velocity memory."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout as original_rollout
from feedback_distribution_pilot import sample
from ensemble_score_objective import energy_costs
from evaluate_checked_particles import horizon_metrics
from train_closed_loop_policy import prefix_windows
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video
from v20_rnn_mixture.engine.features import jets,jet_stencil


def prefix_sequence(full):
    return np.asarray(full[:len(full)//2]).copy()


def initializer_data(base,sequences):
    legacy_types()
    from v19.writers import history_features
    raw=[];target=[];last=[]
    for _,y in sequences:
        # y is already prefix-only; centered labels never cross that boundary.
        v=savgol_filter(y,9,4,deriv=1,axis=0,mode='interp')
        t=np.arange(32,len(y)-4);h=np.stack([y[i-31:i+1] for i in t])
        raw.append(history_features(base,h));target.append(v[t]);last.append(y[t]-y[t-1])
    return np.concatenate(raw),np.concatenate(target),np.concatenate(last)


def fit_initializer(base,sequences):
    legacy_types()
    from v19.writers import ridge
    raw,target,last=initializer_data(base,sequences)
    scale=np.maximum(np.sqrt((raw**2).mean(0)),1e-7)
    coef=ridge(raw/scale,target-last,len(raw)*.001)
    prediction=last+(raw/scale)@coef
    return dict(scale=scale,coef=coef),dict(n=len(raw),penalty=len(raw)*.001,
                proxy_rmse=float(np.sqrt(np.mean((prediction-target)**2))),
                last_difference_proxy_rmse=float(np.sqrt(np.mean((last-target)**2))))


def save_block(path,block):
    np.savez_compressed(path,coef=block['coef'],pair_delta=block['pair_delta'],counts=block['counts'],
        order=block['order'],halfwidth=block['halfwidth'],initializer_scale=block['initializer']['scale'],
        initializer_coef=block['initializer']['coef'])


def load_block(path):
    with np.load(path,allow_pickle=False) as z:
        return dict(type='weak',coef=z['coef'].copy(),pair_delta=z['pair_delta'].copy(),counts=z['counts'].copy(),
                    order=int(z['order']),halfwidth=int(z['halfwidth']),
                    initializer=dict(scale=z['initializer_scale'].copy(),coef=z['initializer_coef'].copy()))


def rollout_memory(s,w,writer,seed,particles=8,trace=False):
    if writer is None:
        r,p,f=original_rollout(s,w,None,seed,particles);return r,p,f,None
    h=np.repeat(w['history'],particles,axis=0).copy();q,mem=s.machine.initialize(h)
    v=writer.initialize(h);dead=~np.isfinite(v).all(1);v[dead]=0
    rng=np.random.default_rng(seed);pred=[];failed=[];records=[];contradictions=0;count=0
    for t in range(w['truth'].shape[1]):
        pe,tr,read=s.machine.read(h,q,mem)
        e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
        with np.errstate(over='ignore',invalid='ignore'):
            y,nv=writer.execute(h,q,r,v)
        dead|=(~np.isfinite(y)).any(1)|(~np.isfinite(nv)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        y[dead]=h[dead,-1];nv[dead]=v[dead];r[dead]=q[dead]
        nh=np.concatenate([h[:,1:],y[:,None]],1);nh[dead]=h[dead]
        hidden=read['read_hidden'].copy();hidden[dead]=mem['hidden'][dead]
        diagnosed=s.base.state_from_history(nh)[0]
        contradictions+=int(((diagnosed!=r)&~dead).sum());count+=int((~dead).sum())
        if trace:records.append(dict(event=e.copy(),source=q.copy(),destination=r.copy(),
                                    velocity=nv.copy(),history=nh.copy(),hidden=hidden.copy()))
        pred.append(y.copy());failed.append(dead.copy());h=nh;q=r;v=nv;mem={'hidden':hidden}
    pred=np.stack(pred,1).reshape(len(w['history']),particles,-1,2)
    failed=np.stack(failed,1).reshape(pred.shape[:-1]);costs=[]
    for t in [50,100,300]:
        if t>pred.shape[2]:continue
        x=pred[:,:,t-1];y=w['truth'][:,t-1]
        cost,_=energy_costs(np.concatenate([np.sin(x),np.cos(x)],-1),np.c_[np.sin(y),np.cos(y)],failed[:,:,t-1]);costs.append(cost.mean())
    horizons=[t for t in [10,25,50,100,300] if t<=pred.shape[2]] or [pred.shape[2]]
    result=dict(seed=seed,objective=float(np.mean(costs)) if costs else None,
        score=horizon_metrics(pred,w['truth'],failed,horizons),destination_contradiction=contradictions/max(count,1),
        per_video={str(video):horizon_metrics(pred[w['video']==video],w['truth'][w['video']==video],failed[w['video']==video],horizons) for video in np.unique(w['video'])})
    return result,pred,failed,records if trace else None


def main():
    root=Path('adaptive_search_results');s=AdaptiveBeam();bridge=LegacyFeatureBridge(s.base);_,Writer=legacy_types()
    from v19.writers import fit_weak
    sequences=[(v,prefix_sequence(load_video(v))) for v in SPLITS['train'][:-3]]
    block=fit_weak(bridge,sequences,halfwidth=8,order=3,local_penalty=3000.)
    initializer,fit_report=fit_initializer(bridge,sequences);block['initializer']=initializer
    path=root/'prefix_velocity_memory_model.npz';save_block(path,block);loaded=load_block(path)
    for key in ['coef','pair_delta','counts']:np.testing.assert_array_equal(block[key],loaded[key])
    for key in initializer:np.testing.assert_array_equal(initializer[key],loaded['initializer'][key])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    holdseq=[(v,prefix_sequence(load_video(v))) for v in SPLITS['train'][-3:]]
    raw,target,last=initializer_data(bridge,holdseq)
    predicted=last+(raw/initializer['scale'])@initializer['coef']
    hold_init=dict(n=len(raw),proxy_rmse=float(np.sqrt(np.mean((predicted-target)**2))),
                   last_difference_proxy_rmse=float(np.sqrt(np.mean((last-target)**2))))
    configs=dict(memory_learned=dict(learned_initialization=True),memory_causal={},
                 reset_causal=dict(reset_velocity=True),memory_shared=dict(learned_initialization=True,local_fraction=0.))
    writers={'original':None,**{name:Writer(bridge,loaded,cfg) for name,cfg in configs.items()}}
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs={name:[] for name in writers}
    for seed in [131017,131018,131019,131020]:
        for name,writer in writers.items():
            r,p,f,_=rollout_memory(s,hold,writer,seed);runs[name].append(r)
            np.savez_compressed(root/f'prefix_velocity_memory_{name}_{seed}.npz',prediction=p,failed=f,
                                truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,r['objective'],r['score']['300']['failure'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(configs=configs,weak_training=block['training'],initializer_fit=fit_report,initializer_hold=hold_init,
                fit_videos=SPLITS['train'][:-3],prefix_lengths={str(v):len(y) for v,y in sequences},
                model_sha256=digest,model_unchanged_during_eval=True,serialization_exact=True,runs=runs,
                mean_objective={k:float(np.mean([r['objective'] for r in rows])) for k,rows in runs.items()},
                note='New prefix-only fit, no legacy writer weights. FixedL8/order3/local3000,initializer ridge N*.001,new protocol not exact old initializer training reproduction. Centered labels only after prefix truncation,causal137features at inference. SameGRU event→r thenchosenwriter,velocity internal. Newwriter fails freeze history/q/hidden/velocity,original runtime delegated unchanged; numericfail not checker validity. TRAINhold only,noDEV/TEST/hyperparameter sweep/defaultpromotion. Proxy velocity is not measured physical velocity.')
    (root/'prefix_velocity_memory_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
