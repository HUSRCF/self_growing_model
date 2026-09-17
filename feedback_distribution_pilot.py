"""Isolated causal-confidence / continuous-writer factorial pilot.

Frozen GRU and F. Fit only TRAIN prefixes, select on whole TRAIN holdout
videos. No checker/search in any rollout arm: isolate the changed controller
and writer before attempting their integration with search. Feedback uses a
previous accepted edge, never a future error or oracle value.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax, logsumexp

from adaptive_search_prototype import AdaptiveBeam
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video, tail_windows, continuous_features
from v20_rnn_mixture.engine.evaluate import metrics


def sample(probs, uniform):
    return np.minimum((probs.cumsum(-1) < uniform[:, None]).sum(-1), probs.shape[-1]-1)


def edge_feedback(s, h, q, y, pe, tr, e, r):
    """Available after accepting y; describes the preceding decision only."""
    ids = np.arange(len(q))
    pq = np.einsum('be,ber->br', pe, tr)
    margin = s.checker.score(h[:, :-1], h[:, -1], y, q)
    return np.c_[np.log(np.maximum(pe[ids,e],1e-12)),
                 np.log(np.maximum(tr[ids,e,r],1e-12)),
                 np.log(np.maximum(pq[ids,r],1e-12)),
                 -(pe*np.log(np.maximum(pe,1e-12))).sum(1),
                 -(pq*np.log(np.maximum(pq,1e-12))).sum(1),
                 np.minimum(margin,100.), margin > s.checker.threshold[q]]


def inputs(s, h, q, hidden, pe, tr):
    return np.c_[s.machine.head.transform(continuous_features(h,s.base)),
                 np.eye(s.base.k)[q], hidden,
                 np.log(np.maximum(np.einsum('be,ber->br',pe,tr),1e-12))]


def observed_data(s, split, windows=16, steps=64):
    """Teacher-forced chains, with previous-edge statistics shifted by one."""
    rng=np.random.default_rng(92017)
    out={k:[] for k in ['x','feedback','pe','tr','q','r','residual','video']}
    if split!='train':out['residual_all']=[]
    for video in SPLITS[split]:
        full=load_video(video)
        y=full[:len(full)//2] if split=='train' else full[len(full)//2:]
        starts=np.linspace(63,len(y)-steps-1,windows,dtype=int)
        h=np.stack([y[t-31:t+1] for t in starts])
        q,mem=s.machine.initialize(h); hidden=mem['hidden']; previous=np.zeros((len(h),7))
        for t in range(steps):
            pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
            nxt=y[starts+t+1]
            nh=np.concatenate([h[:,1:],nxt[:,None]],1)
            r=s.base.state_from_history(nh)[0]
            center=s.base.execute_rule(h,q,r)
            # e is latent in observed data. Sample its posterior conditional
            # on observed r; this feedback is used ONLY at the following step.
            post=pe*np.take_along_axis(tr,r[:,None,None],axis=2)[:,:,0]
            post/=post.sum(1,keepdims=True)
            e=sample(post,rng.random(len(h)))
            values=dict(x=inputs(s,h,q,hidden,pe,tr),feedback=previous.copy(),
                        pe=pe,tr=tr,q=q.copy(),r=r.copy(),residual=nxt-center,
                        video=np.full(len(h),video))
            if split!='train':
                all_centers=s.base.execute_rule(np.repeat(h,s.base.k,axis=0),
                                                np.repeat(q,s.base.k),np.tile(np.arange(s.base.k),len(h)))
                values['residual_all']=nxt[:,None]-all_centers.reshape(len(h),s.base.k,2)
            for k,v in values.items():out[k].append(v)
            previous=edge_feedback(s,h,q,nxt,pe,tr,e,r)
            h,q,hidden=nh,r,read['read_hidden']
        print(f'observed {split} video {video}',flush=True)
    return {k:np.concatenate(v) for k,v in out.items()}


def fit_adapter(data, use_feedback, epochs=100):
    # Same width and parameter count; control receives seven constant zeros.
    torch.manual_seed(1901); torch.set_num_threads(1)
    x=np.c_[data['x'],data['feedback'] if use_feedback else np.zeros_like(data['feedback'])]
    hold=np.isin(data['video'],SPLITS['train'][-3:]); fit=~hold
    mean=x[fit].mean(0); scale=np.maximum(x[fit].std(0),1e-5)
    z=torch.tensor(np.clip((x-mean)/scale,-8,8),dtype=torch.float32)
    logpe=torch.tensor(np.log(np.maximum(data['pe'],1e-12)),dtype=torch.float32)
    logtr=torch.tensor(np.log(np.maximum(data['tr'],1e-12)),dtype=torch.float32)
    labels=torch.tensor(data['r'],dtype=torch.long)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,8))
    torch.nn.init.zeros_(model[-1].weight); torch.nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    def nll(mask):
        correction=model(z[mask])
        # Residual q logits are applied INSIDE each event's transition law.
        conditional=torch.log_softmax(logtr[mask]+correction[:,None,:],dim=-1)
        marginal=torch.logsumexp(logpe[mask,:,None]+conditional,dim=1)
        return torch.nn.functional.nll_loss(marginal,labels[mask])
    best=float('inf'); state=None; epoch_best=0; trace=[]
    for epoch in range(epochs+1):
        if epoch%5==0:
            with torch.no_grad():val=float(nll(hold))
            trace.append(dict(epoch=epoch,holdout_nll=val))
            if val<best:
                best=val;epoch_best=epoch
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch==epochs:break
        loss=nll(fit)
        optimizer.zero_grad();loss.backward();optimizer.step()
    a=dict(mean=mean,scale=scale,w1=state['0.weight'].numpy().T,
           b1=state['0.bias'].numpy(),w2=state['2.weight'].numpy().T,b2=state['2.bias'].numpy())
    return a,dict(feedback=use_feedback,selected_epoch=epoch_best,holdout_nll=best,trace=trace)


def corrected_tr(a,x,feedback,tr,use_feedback):
    if a is None:return tr
    z=np.c_[x,feedback if use_feedback else np.zeros_like(feedback)]
    z=np.clip((z-a['mean'])/a['scale'],-8,8)
    delta=np.tanh(z@a['w1']+a['b1'])@a['w2']+a['b2']
    return softmax(np.log(np.maximum(tr,1e-12))+delta[:,None,:],axis=-1)


def fit_covariance(data):
    fit=~np.isin(data['video'],SPLITS['train'][-3:])
    residual=data['residual'][fit]; qs=data['q'][fit]; rs=data['r'][fit]
    # Fixed center F: use second moment about ZERO, not a silently shifted mean.
    global_cov=residual.T@residual/len(residual)+np.eye(2)*1e-10
    covariance=np.empty((8,8,2,2)); counts=np.zeros((8,8),dtype=int)
    for q in range(8):
        for r in range(8):
            a=residual[(qs==q)&(rs==r)]; counts[q,r]=len(a)
            covariance[q,r]=(a.T@a+32*global_cov)/(len(a)+32)+np.eye(2)*1e-10
    return covariance,counts


def gaussian_audit(data,cov):
    c=cov[data['q'],data['r']]; err=data['residual']
    maha=np.einsum('bi,bij,bj->b',err,np.linalg.inv(c),err)
    nll=.5*(2*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+maha)
    return dict(conditional_on_observed_q_r=True,n_samples=len(err),
                nll=float(nll.mean()),coverage90_ellipse=float((maha<=4.605170186).mean()),
                mean_center_error=err.mean(0).tolist(),
                note='Conditional local residual diagnostic, not marginal predictive likelihood or rollout coverage.')


def residual_time_audit(data,cov,windows=16,steps=64):
    """Observed standardized residual lag-one correlation, not a noise model."""
    left=[];right=[]
    for v in SPLITS['train'][:-3]:
        mask=data['video']==v
        c=np.linalg.cholesky(cov[data['q'][mask],data['r'][mask]])
        z=np.linalg.solve(c,data['residual'][mask,:,None])[:,:,0].reshape(steps,windows,2)
        left.append(z[:-1].reshape(-1,2));right.append(z[1:].reshape(-1,2))
    a=np.concatenate(left);b=np.concatenate(right)
    return dict(standardized_lag1_correlation=[float(np.corrcoef(a[:,j],b[:,j])[0,1]) for j in range(2)],
                note='Overlapping observed-history chains on fit videos; diagnostic only, not independent samples.')


def rollout_arm(s,windows,a,cov,use_feedback,continuous,seed,particles=8,rho=None):
    n=len(windows['history']); steps=windows['truth'].shape[1]
    h=np.repeat(windows['history'],particles,axis=0)
    q,mem=s.machine.initialize(h); hidden=mem['hidden']; previous=np.zeros((len(h),7))
    # Separate streams: adding Gaussian samples never shifts event/q uniforms.
    rng=np.random.default_rng(seed); noise_rng=np.random.default_rng(seed+100000)
    chol=np.linalg.cholesky(cov)
    previous_noise=np.zeros((len(h),2))
    pred=np.empty((len(h),steps,2));failed=np.zeros((len(h),steps),dtype=bool)
    dead=np.zeros(len(h),dtype=bool); rejected=0; checked=0
    for t in range(steps):
        pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
        tr=corrected_tr(a,inputs(s,h,q,hidden,pe,tr),previous,tr,use_feedback)
        ids=np.arange(len(h)); e=sample(pe,rng.random(len(h)))
        r=sample(tr[ids,e],rng.random(len(h)))
        y=s.base.execute_rule(h,q,r)
        if continuous:
            noise=noise_rng.normal(size=(len(h),2))
            if rho is not None and t>0:
                noise=rho*previous_noise+np.sqrt(1-rho*rho)*noise
            previous_noise=noise.copy()
            y+=np.einsum('bij,bj->bi',chol[q,r],noise)
        # Fixed numeric safety guard in all arms; never retries or uses truth.
        bad=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        dead|=bad; y[dead]=h[dead,-1]
        fb=edge_feedback(s,h,q,y,pe,tr,e,r)
        if t>0:
            rejected+=int(fb[~dead,-1].sum());checked+=int((~dead).sum())
        pred[:,t]=y;failed[:,t]=dead
        previous=fb;h=np.concatenate([h[:,1:],y[:,None]],1)
        q,hidden=r,read['read_hidden']
    pred=pred.reshape(n,particles,steps,2);failed=failed.reshape(n,particles,steps)
    score,_=metrics(pred,windows['truth'],failed)
    per_video={str(v):metrics(pred[windows['video']==v],windows['truth'][windows['video']==v],
                             failed[windows['video']==v])[0] for v in SPLITS['dev']}
    return dict(seed=seed,score=score,per_video=per_video,
                diagnostic_mixture_rejection_rate=rejected/max(checked,1)),pred,failed


def temporal_extension(args,s):
    out=Path(args.output);report=json.loads(out.read_text())
    cov=np.load(out.with_name(out.stem+'_covariance.npz'))['covariance']
    rho=np.clip(report['fit_residual_time_audit']['standardized_lag1_correlation'],-.95,.95)
    report['temporal_extension']=dict(rho=rho.tolist(),
        note='Post-hoc mechanistic DEV ablation. Coefficients estimated on fit TRAIN only; no DEV rho sweep. '
             'AR standardized noise has zero stationary mean but nonzero history-conditional mean. '
             'Thus this extension no longer fixes every conditional center exactly at F.')
    windows=tail_windows('dev',300,args.per_video);start=time.perf_counter()
    for key in ['frozen','adapter','feedback']:
        a=None if key=='frozen' else dict(np.load(out.with_name(out.stem+'_'+key+'.npz')))
        name=key+'_ar_gaussian';report['arms'][name]=[]
        for seed in args.seeds:
            run,p,f=rollout_arm(s,windows,a,cov,key=='feedback',True,seed,rho=rho)
            report['arms'][name].append(run)
            np.savez_compressed(out.with_name(out.stem+f'_{name}_{seed}.npz'),
                                prediction=p,failed=f,truth=windows['truth'],
                                video=windows['video'],window_start=windows['start'])
            print(name,seed,{t:round(run['score'][str(t)]['embedding_rmse'],6) for t in [50,100,300]},flush=True)
        report['temporal_extension']['seconds']=time.perf_counter()-start
        out.write_text(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--per-video',type=int,default=8)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1729,2718,3141])
    parser.add_argument('--output',default='adaptive_search_results/feedback_distribution_pilot.json')
    parser.add_argument('--temporal-only',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.parent.mkdir(exist_ok=True)
    torch.set_num_threads(1);start=time.perf_counter();s=AdaptiveBeam()
    if args.temporal_only:
        temporal_extension(args,s)
        return
    data=observed_data(s,'train'); cov,counts=fit_covariance(data)
    models={};training={}
    for key,feedback in [('adapter',False),('feedback',True)]:
        models[key],training[key]=fit_adapter(data,feedback)
        np.savez(out.with_name(out.stem+'_'+key+'.npz'),**models[key])
        print(key,training[key]['selected_epoch'],training[key]['holdout_nll'],flush=True)
    np.savez(out.with_name(out.stem+'_covariance.npz'),covariance=cov,counts=counts)
    hold=np.isin(data['video'],SPLITS['train'][-3:])
    audits={'train_holdout':gaussian_audit({k:v[hold] for k,v in data.items()},cov)}
    dev=observed_data(s,'dev',windows=8)
    audits['dev']=gaussian_audit(dev,cov)
    dev_nll={}
    marginal_nll={}
    for key,a in [('frozen',None),*models.items()]:
        tr=corrected_tr(a,dev['x'],dev['feedback'],dev['tr'],key=='feedback')
        prob=np.einsum('be,ber->br',dev['pe'],tr)
        dev_nll[key]=float(-np.log(np.maximum(prob[np.arange(len(prob)),dev['r']],1e-12)).mean())
        c=cov[dev['q']];err=dev['residual_all']
        maha=np.einsum('bri,brij,brj->br',err,np.linalg.inv(c),err)
        logdensity=-.5*(2*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+maha)
        marginal_nll[key]=float(-logsumexp(np.log(np.maximum(prob,1e-12))+logdensity,axis=1).mean())
    report=dict(config=vars(args),training=training,covariance_counts=counts.tolist(),
                conditional_gaussian=audits,dev_observed_q_nll=dev_nll,
                dev_gaussian_marginal_angle_nll=marginal_nll,
                fit_residual_time_audit=residual_time_audit(data,cov),
                fit_videos=SPLITS['train'][:-3],selection_videos=SPLITS['train'][-3:],
                n_training_states=len(data['q']),arms={},
                limitations=['DEV exploratory, not blind test.',
                  'No checking, lookahead or rollback in any arm; not directly the checked-search deployment.',
                  'Teacher-forced fitting has exposure mismatch; previous latent event sampled conditional on observed r.',
                  'Confidence features are policy beliefs, not calibrated expected future rewards.',
                  'Gaussian residuals are temporally independent and centered exactly at F; not a VAE.',
                  'Frozen backbone already saw the TRAIN holdout videos; holdout is for new adapters only.',
                  'Same-capacity control zeros feedback channels; does not separate extra history from confidence semantics.'])
    windows=tail_windows('dev',300,args.per_video)
    for key,a in [('frozen',None),*models.items()]:
        for continuous in [False,True]:
            name=key+('_gaussian' if continuous else '_point');report['arms'][name]=[]
            for seed in args.seeds:
                run,p,f=rollout_arm(s,windows,a,cov,key=='feedback',continuous,seed)
                report['arms'][name].append(run)
                np.savez_compressed(out.with_name(out.stem+f'_{name}_{seed}.npz'),
                                    prediction=p,failed=f,truth=windows['truth'],
                                    video=windows['video'],window_start=windows['start'])
                print(name,seed,{t:round(run['score'][str(t)]['embedding_rmse'],6) for t in [50,100,300]},flush=True)
            report['seconds']=time.perf_counter()-start
            out.write_text(json.dumps(report,indent=2))
    print('saved',out,flush=True)


if __name__=='__main__':main()
