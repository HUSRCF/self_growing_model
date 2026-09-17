"""On-policy multi-step policy-gradient pilot, with frozen GRU and F.

TRAIN prefixes only; three TRAIN videos select checkpoints. Original frozen
policy is epoch zero. No numerical differentiation through the environment:
REINFORCE differentiates the sampled transition log probability. Frozen
event probabilities remain operational and choose e before r. Feedback
describes frozen-backbone beliefs, so no omitted differentiable dependency
through previous learned-policy confidence features is introduced.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adaptive_search_prototype import AdaptiveBeam
from feedback_distribution_pilot import inputs, edge_feedback, sample, corrected_tr
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video, tail_windows
from v20_rnn_mixture.engine.evaluate import metrics


def prefix_windows(videos,steps=100,per_video=4):
    result={k:[] for k in ['history','truth','video','start']}
    for video in videos:
        full=load_video(video);y=full[:len(full)//2]
        starts=np.linspace(63,len(y)-steps-1,per_video,dtype=int)
        for t in starts:
            result['history'].append(y[t-31:t+1]);result['truth'].append(y[t+1:t+steps+1])
            result['video'].append(video);result['start'].append(t)
    return {k:np.asarray(v) for k,v in result.items()}


def leave_one_out(values,particles):
    """Other trajectories, not this trajectory, supply a baseline."""
    a=values.reshape(-1,particles)
    return ((a.sum(1,keepdims=True)-a)/(particles-1)).reshape(-1)


def future_costs(costs,horizons,steps):
    # Costs at already-passed horizons must not credit later actions.
    return np.stack([sum(c for horizon,c in zip(horizons,costs) if horizon>t)/len(horizons)
                     for t in range(steps)])


class Actor:
    def __init__(self,s,use_feedback,seed=1901):
        torch.manual_seed(seed)
        d=len(s.machine.head.a['mean'])+8+s.machine.head.width+8+7
        self.model=torch.nn.Sequential(torch.nn.Linear(d,32),torch.nn.Tanh(),torch.nn.Linear(32,8))
        torch.nn.init.zeros_(self.model[-1].weight);torch.nn.init.zeros_(self.model[-1].bias)
        self.use_feedback=use_feedback
        # Fixed, explicit scaling; no DEV-derived feature statistics.
        self.mean=np.zeros(d);self.scale=np.ones(d)
        self.scale[-15:-7]=5.
        self.scale[-7:]=[5.,5.,5.,2.,2.,10.,1.]

    def transformed(self,x,feedback):
        all_x=np.c_[x,feedback if self.use_feedback else np.zeros_like(feedback)]
        return np.clip((all_x-self.mean)/self.scale,-8,8)

    def arrays(self):
        a=self.model.state_dict()
        return dict(mean=self.mean.copy(),scale=self.scale.copy(),
                    w1=a['0.weight'].detach().numpy().T.copy(),b1=a['0.bias'].detach().numpy().copy(),
                    w2=a['2.weight'].detach().numpy().T.copy(),b2=a['2.bias'].detach().numpy().copy())


def run_policy(s,windows,actor=None,arrays=None,use_feedback=False,seed=1729,particles=4,training=False,objective_kind='mse'):
    if objective_kind not in ('mse','energy_u'):raise ValueError('Unknown objective')
    n=len(windows['history']);steps=windows['truth'].shape[1]
    h=np.repeat(windows['history'],particles,axis=0)
    truth=np.repeat(windows['truth'],particles,axis=0)
    q,mem=s.machine.initialize(h);hidden=mem['hidden'];previous=np.zeros((len(h),7))
    rng=np.random.default_rng(seed);ids=np.arange(len(h))
    predictions=[];failures=[];log_probs=[];regularizers=[]
    dead=np.zeros(len(h),dtype=bool)
    horizons=[t for t in [50,100] if t<=steps]
    errors=[];energy_advantages=[];rejected=0;checked=0
    for t in range(steps):
        pe,raw_tr,read=s.machine.read(h,q,{'hidden':hidden})
        x=inputs(s,h,q,hidden,pe,raw_tr)
        if training:
            z=torch.tensor(actor.transformed(x,previous),dtype=torch.float32)
            delta=actor.model(z)
            raw=torch.tensor(np.log(np.maximum(raw_tr,1e-12)),dtype=torch.float32)
            logtr=torch.log_softmax(raw+delta[:,None,:],dim=-1)
            tr=logtr.exp().detach().numpy().astype(float)
            tr/=tr.sum(-1,keepdims=True)
        else:
            tr=corrected_tr(arrays,x,previous,raw_tr,use_feedback)
        e=sample(pe,rng.random(len(h)));r=sample(tr[ids,e],rng.random(len(h)))
        if training:
            log_probs.append(logtr[ids,e,r])
            # KL(new transition || frozen transition), averaged over frozen e.
            regularizers.append((logtr.exp()*(logtr-raw)*torch.tensor(pe[:,:,None],dtype=torch.float32)).sum((1,2)).mean())
        y=s.base.execute_rule(h,q,r)
        dead|=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        y[dead]=h[dead,-1]
        # Crucially RAW (frozen) tr, not the trainable actor's tr, is fed back.
        previous=edge_feedback(s,h,q,y,pe,raw_tr,e,r)
        if t>0:
            rejected+=int(previous[~dead,-1].sum());checked+=int((~dead).sum())
        predictions.append(y.copy());failures.append(dead.copy())
        if t+1 in horizons:
            embed=np.c_[np.sin(y),np.cos(y)]
            target=np.c_[np.sin(truth[:,t]),np.cos(truth[:,t])]
            if objective_kind=='mse':
                errors.append(((embed-target)**2).mean(1)+2*dead)
            else:
                from ensemble_score_objective import energy_costs
                cost,baseline=energy_costs(embed.reshape(n,particles,-1),target.reshape(n,particles,-1)[:,0],dead.reshape(n,particles))
                errors.append(np.repeat(cost,particles))
                energy_advantages.append((particles*(cost[:,None]-baseline)).reshape(-1))
        h=np.concatenate([h[:,1:],y[:,None]],1);q=r;hidden=read['read_hidden']
    pred=np.stack(predictions,1).reshape(n,particles,steps,2)
    failed=np.stack(failures,1).reshape(n,particles,steps)
    objective=float(np.mean(errors))
    if training:
        if particles<2:raise ValueError('Leave-one-out baseline requires multiple particles')
        if objective_kind=='mse':
            returns=future_costs(errors,horizons,steps)
            baseline=np.stack([leave_one_out(v,particles) for v in returns])
            advantage=returns-baseline
        else:
            advantage=future_costs(energy_advantages,horizons,steps)
        # Normalizing by a random batch statistic would alter the estimator.
        # Fixed scale changes only effective learning rate, not credit signs.
        weights=torch.tensor(advantage/.02,dtype=torch.float32)
        policy_loss=(torch.stack(log_probs)*weights).sum(0).mean()/steps
        return policy_loss,torch.stack(regularizers).mean(),dict(objective=objective,
                     advantage_std=float(advantage.std()),failure=float(failed[:,:,-1].mean()))
    score,_=metrics(pred,windows['truth'],failed)
    return dict(objective=objective,seed=seed,score=score,
                diagnostic_mixture_rejection_rate=rejected/max(checked,1),
                per_video={str(v):metrics(pred[windows['video']==v],windows['truth'][windows['video']==v],
                                         failed[windows['video']==v])[0] for v in np.unique(windows['video'])}),pred,failed


def train_one(s,args,use_feedback):
    objective_kind=getattr(args,'objective_kind','mse')
    selection_kind=getattr(args,'selection_objective_kind',None) or objective_kind
    actor=Actor(s,use_feedback,args.train_seed);optimizer=torch.optim.AdamW(actor.model.parameters(),lr=.003,weight_decay=.01)
    fit=prefix_windows(SPLITS['train'][:-3],per_video=4)
    sampling=getattr(args,'window_sampling','fixed')
    pool=None
    if sampling!='fixed':
        from training_window_sampler import TrainingPrefixPool
        pool=TrainingPrefixPool(s.base)
    hold=prefix_windows(SPLITS['train'][-3:],per_video=8)
    best=float('inf');chosen=None;chosen_epoch=0;trace=[];train_trace=[];start=time.perf_counter()
    checkpoints=set(np.linspace(0,args.epochs,7,dtype=int))
    for epoch in range(args.epochs+1):
        if epoch in checkpoints:
            a=actor.arrays()
            checks=[run_policy(s,hold,arrays=a,use_feedback=use_feedback,seed=seed,particles=8,objective_kind=selection_kind)[0]
                    for seed in [12017,12019]]
            objective=float(np.mean([c['objective'] for c in checks]))
            trace.append(dict(epoch=epoch,holdout_objective=objective,
                              holdout_100_rmse=float(np.mean([c['score']['100']['embedding_rmse'] for c in checks]))))
            if objective<best:best=objective;chosen=a;chosen_epoch=epoch
            print('feedback',use_feedback,'epoch',epoch,'holdout',objective,'best',chosen_epoch,flush=True)
        if epoch==args.epochs:break
        optimizer.zero_grad()
        batch_stats=[]
        for batch in range(args.accumulate):
            seed=42000+epoch*args.accumulate+batch+args.train_seed-1901
            if pool is not None:fit=pool.sample(seed+15000,mode=sampling)
            loss,kl,stats=run_policy(s,fit,actor=actor,training=True,seed=seed,particles=4,objective_kind=objective_kind)
            ((loss+args.kl_weight*kl)/args.accumulate).backward()
            batch_stats.append(dict(loss=float(loss.detach()),kl=float(kl.detach()),**stats))
        grad_norm=float(torch.nn.utils.clip_grad_norm_(actor.model.parameters(),1.))
        optimizer.step()
        averaged={k:float(np.mean([b[k] for b in batch_stats])) for k in batch_stats[0]}
        train_trace.append(dict(epoch=epoch+1,batches_seen=(epoch+1)*args.accumulate,gradient_norm=grad_norm,**averaged))
        if epoch%5==0:print('train',train_trace[-1],flush=True)
    return chosen,dict(selected_epoch=chosen_epoch,holdout_objective=best,trace=trace,train_trace=train_trace,
                       sampler_audit=pool.audit() if pool else None,seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--kl-weight',type=float,default=.01)
    parser.add_argument('--train-seed',type=int,default=1901)
    parser.add_argument('--accumulate',type=int,default=1)
    parser.add_argument('--window-sampling',choices=['fixed','uniform','stratified'],default='fixed')
    parser.add_argument('--objective-kind',choices=['mse','energy_u'],default='mse')
    parser.add_argument('--selection-objective-kind',choices=['mse','energy_u'],default=None)
    parser.add_argument('--output',default='adaptive_search_results/closed_loop_policy.json')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.accumulate<1 or args.epochs<1:raise ValueError('Positive accumulation and update counts required')
    s=AdaptiveBeam();out=Path(args.output);out.parent.mkdir(exist_ok=True);start=time.perf_counter()
    report=dict(config=vars(args),fit_videos=SPLITS['train'][:-3],selection_videos=SPLITS['train'][-3:],training={},arms={},
      limitations=['On-policy score-function estimator; F and GRU frozen. No checker/search/noise.',
        'Training/selection objectives at50/100: '+args.objective_kind+'/'+(args.selection_objective_kind or args.objective_kind)+'; not ensemble-mean RMSE. Energy uses off-diagonal U score and whole-trajectory leave-out baselines.',
        'Same fixed gradient scale/learning rate across objectives; loss units and clipping incidence can differ. This is not an equal-gradient-magnitude comparison.',
        '3s rollout is extrapolation beyond100-step training horizon.',
        'Previous confidence uses frozen-backbone probabilities; not the actor probabilities from earlier pilot.',
        'Original backbone already saw all TRAIN videos. Holdout is only for the new adapter.',
        'One optimizer seed; repeated trajectory seeds do not establish training-seed robustness.'])
    models={'frozen':None}
    for name,feedback in [('closed_loop',False),('closed_loop_feedback',True)]:
        models[name],report['training'][name]=train_one(s,args,feedback)
        np.savez(out.with_name(out.stem+'_'+name+'.npz'),**models[name])
        out.write_text(json.dumps(report,indent=2))
    windows=tail_windows('dev',300,8)
    for name,a in models.items():
        report['arms'][name]=[]
        for seed in [1729,2718,3141]:
            r,p,f=run_policy(s,windows,arrays=a,use_feedback=name.endswith('feedback'),seed=seed,particles=8,objective_kind=args.objective_kind)
            report['arms'][name].append(r)
            np.savez_compressed(out.with_name(out.stem+'_'+name+f'_{seed}.npz'),prediction=p,failed=f,
                                truth=windows['truth'],video=windows['video'],window_start=windows['start'])
            print(name,seed,{t:r['score'][str(t)]['embedding_rmse'] for t in [50,100,300]},flush=True)
        report['seconds']=time.perf_counter()-start;out.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
