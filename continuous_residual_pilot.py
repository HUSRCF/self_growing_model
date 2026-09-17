"""Causal bounded center correction; teacher-forced fit, free-run TRAIN gate."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from feedback_distribution_pilot import sample
from train_closed_loop_policy import prefix_windows,run_policy
from ensemble_score_objective import energy_costs
from evaluate_checked_particles import horizon_metrics
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import continuous_features,load_video,tail_windows


def design(raw,q,r,mean,scale):
    return np.c_[np.ones(len(raw)),np.clip((raw-mean)/scale,-8,8),np.eye(8)[q],np.eye(8)[r]]


def correction(model,base,h,q,r):
    if model is None:return np.zeros((len(h),2))
    if model['kind']=='constant':return np.broadcast_to(model['value'],(len(h),2)).copy()
    if model['kind']=='velocity':
        return model['value']*np.tanh((h[:,-1]-h[:,-2])/model['velocity_scale'])
    x=design(continuous_features(h,base),q,r,model['mean'],model['scale'])
    return np.clip(x@model['coef'],-model['cap'],model['cap'])


def collect(s):
    rows={k:[] for k in ['history','q','truth','probability','video']}
    for v in SPLITS['train']:
        full=load_video(v);y=full[:len(full)//2];starts=np.linspace(63,len(y)-102,16,dtype=int)
        h=np.stack([y[t-31:t+1] for t in starts]);q,mem=s.machine.initialize(h)
        for t in range(64):
            pe,tr,read=s.machine.read(h,q,mem);truth=y[starts+t+1]
            for k,a in dict(history=h.copy(),q=q.copy(),truth=truth,probability=np.einsum('be,ber->br',pe,tr),video=np.full(len(h),v)).items():rows[k].append(a)
            h=np.concatenate([h[:,1:],truth[:,None]],1);q=s.base.state_from_history(h)[0];mem={'hidden':read['read_hidden']}
    return {k:np.concatenate(v) for k,v in rows.items()}


def fit_models(base,data):
    h=data['history'];q=data['q'];n=len(h);fit=np.isin(data['video'],SPLITS['train'][:-3])
    centers=base.execute_rule(np.repeat(h,8,axis=0),np.repeat(q,8),np.tile(np.arange(8),n)).reshape(n,8,2)
    raw=continuous_features(h,base);mean=raw[fit].mean(0);scale=np.maximum(raw[fit].std(0),1e-5)
    residual=np.angle(np.exp(1j*(data['truth'][:,None]-centers)))
    cap=np.quantile(np.abs(residual[fit]),.99,axis=(0,1))
    x=design(np.repeat(raw[fit],8,axis=0),np.repeat(q[fit],8),np.tile(np.arange(8),fit.sum()),mean,scale)
    w=data['probability'][fit].reshape(-1);target=residual[fit].reshape(-1,2)
    gram=x.T@(w[:,None]*x)/w.sum();rhs=x.T@(w[:,None]*target)/w.sum()
    models={'zero':None,'constant':dict(kind='constant',value=(w[:,None]*target).sum(0)/w.sum())}
    for alpha in [1e-4,.01,1.]:
        penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
        models[f'ridge_{alpha}']=dict(kind='ridge',mean=mean,scale=scale,cap=cap,coef=np.linalg.solve(gram+penalty,rhs))
    return models,centers,fit


def observed_score(s,data,centers,model,mask):
    h=data['history'][mask];q=data['q'][mask];p=data['probability'][mask];n=len(h)
    rr=np.tile(np.arange(8),n);hh=np.repeat(h,8,axis=0);qq=np.repeat(q,8)
    delta=correction(model,s.base,hh,qq,rr).reshape(n,8,2);pred=centers[mask]+delta
    emb=np.concatenate([np.sin(pred),np.cos(pred)],-1);y=data['truth'][mask];target=np.c_[np.sin(y),np.cos(y)]
    errors=((emb-target[:,None])**2).mean(-1)
    next_h=np.concatenate([hh[:,1:],pred.reshape(-1,1,2)],1)
    diagnosed=s.base.state_from_history(next_h)[0].reshape(n,8)
    reject,_=s.checker.reject(hh[:,:-1],hh[:,-1],pred.reshape(-1,2),qq)
    return dict(expected_embedding_mse=float((p*errors).sum(1).mean()),
                destination_contradiction=float((p*(diagnosed!=np.arange(8)[None])).sum(1).mean()),
                current_middle_rejection=float((p*reject.reshape(n,8)).sum(1).mean()),
                correction_rms_rad=float(np.sqrt((p[:,:,None]*delta**2).sum((1,2)).mean()/2)))


def rollout(s,w,model,seed,particles=8,first_noise=None):
    h=np.repeat(w['history'],particles,axis=0);q,mem=s.machine.initialize(h);rng=np.random.default_rng(seed)
    if first_noise is not None:
        first_noise=np.asarray(first_noise)
        if first_noise.shape!=(len(h),2) or not np.isfinite(first_noise).all():
            raise ValueError('finite first_noise[window*particle,2] required')
    dead=np.zeros(len(h),bool);pred=[];failed=[];contradictions=0;rejections=0;count=0
    for t in range(w['truth'].shape[1]):
        pe,tr,read=s.machine.read(h,q,mem)
        e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
        y=s.base.execute_rule(h,q,r)
        if model is not None:y=y+correction(model,s.base,h,q,r)
        if t==0 and first_noise is not None:y=y+first_noise
        dead|=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1);y[dead]=h[dead,-1]
        nh=np.concatenate([h[:,1:],y[:,None]],1)
        diagnosed=s.base.state_from_history(nh)[0]
        contradictions+=int(((diagnosed!=r)&~dead).sum());count+=int((~dead).sum())
        if t>0:rejections+=int((s.checker.reject(h[:,:-1],h[:,-1],y,q)[0]&~dead).sum())
        pred.append(y.copy());failed.append(dead.copy());h=nh;q=r;mem={'hidden':read['read_hidden']}
    pred=np.stack(pred,1).reshape(len(w['history']),particles,-1,2);failed=np.stack(failed,1).reshape(pred.shape[:-1])
    costs=[]
    for t in [50,100,300]:
        if t>pred.shape[2]:continue
        x=pred[:,:,t-1];y=w['truth'][:,t-1]
        cost,_=energy_costs(np.concatenate([np.sin(x),np.cos(x)],-1),np.c_[np.sin(y),np.cos(y)],failed[:,:,t-1]);costs.append(cost.mean())
    return dict(seed=seed,objective=float(np.mean(costs)),score=horizon_metrics(pred,w['truth'],failed),
                destination_contradiction=contradictions/max(count,1),diagnostic_rejections=rejections,
                per_video={str(v):horizon_metrics(pred[w['video']==v],w['truth'][w['video']==v],failed[w['video']==v]) for v in np.unique(w['video'])}),pred,failed


def main():
    s=AdaptiveBeam();root=Path('adaptive_search_results');data=collect(s);models,centers,fit=fit_models(s.base,data)
    model_files={}
    for name,model in models.items():
        if model is not None:
            path=root/f'continuous_residual_model_{name}.npz';np.savez(path,**model);model_files[name]=path.name
            with np.load(path,allow_pickle=False) as stored:
                for k,v in model.items():np.testing.assert_array_equal(v,stored[k])
    observed={name:{'fit':observed_score(s,data,centers,model,fit),'hold':observed_score(s,data,centers,model,~fit)} for name,model in models.items()}
    chosen=min([name for name in models if name.startswith('ridge')],key=lambda name:observed[name]['hold']['expected_embedding_mse'])
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);free={}
    for name in ['zero','constant',chosen]:
        free[name]=[]
        for seed in [12017,12019]:
            r,p,f=rollout(s,hold,models[name],seed);free[name].append(r)
            if name=='zero':
                _,old,oldfail=run_policy(s,hold,seed=seed,particles=8)
                np.testing.assert_array_equal(p,old);np.testing.assert_array_equal(f,oldfail)
            np.savez_compressed(root/f'continuous_residual_hold_{name}_{seed}.npz',prediction=p,failed=f,truth=hold['truth'],video=hold['video'],window_start=hold['start'])
        print(name,observed[name]['hold'],'free cost',np.mean([r['objective'] for r in free[name]]),flush=True)
    winner=min(free,key=lambda name:np.mean([r['objective'] for r in free[name]]))
    report=dict(observed=observed,selected_ridge=chosen,holdout=free,selected_by_free_holdout=winner,
                zero_free_parity=True,model_files=model_files,model_serialization_verified=True,
                constant_correction_rad=models['constant']['value'].tolist(),ridge_cap_rad=models[chosen]['cap'].tolist(),
                fit_rows=int(fit.sum()),hold_rows=int((~fit).sum()),dev={})
    np.savez(root/'continuous_residual_selected_ridge.npz',**models[chosen])
    if winner!='zero':
        dev=tail_windows('dev',300,8)
        for name in ['zero',winner]:
            report['dev'][name]=[]
            for seed in [1729,2718,3141]:
                r,p,f=rollout(s,dev,models[name],seed);report['dev'][name].append(r)
                if name=='zero':
                    with np.load(root/f'feedback_distribution_pilot_frozen_point_{seed}.npz') as original:
                        np.testing.assert_array_equal(p,original['prediction']);np.testing.assert_array_equal(f,original['failed'])
                np.savez_compressed(root/f'continuous_residual_dev_{name}_{seed}.npz',prediction=p,failed=f,truth=dev['truth'],video=dev['video'],window_start=dev['start'])
    report['note']='Deterministic weighted ridge fit on10TRAIN-prefix videos, three TRAIN videos select ridge by one-step risk then gate free rollout. Same selection videos reused across stages, not independent confirmation. No oracle destination labels, no noise, unchanged event/q/F order with additive bounded correction after F. Baseline remains eligible. No checker rejection is enforced; diagnostics only. DEV evaluated only if nonzero free-holdout winner. No TEST or frozen file changes.'
    (root/'continuous_residual_pilot.json').write_text(json.dumps(report,indent=2));print('winner',winner,flush=True)


if __name__=='__main__':main()
