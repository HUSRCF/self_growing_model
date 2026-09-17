"""Pairwise candidate ranking on generated TRAIN-prefix states.

Select checkpoints using three held-out TRAIN videos; no dev/test fitting.
All candidate labels share a fixed greedy continuation policy, not an oracle.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from train_search_value import features
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def step(s,h,q,hidden,rng=None):
    pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
    joint=(pe[:,:,None]*tr).reshape(len(h),-1)
    if rng is None:
        a=joint.argmax(1)
    else:
        a=np.minimum((joint.cumsum(1)<rng.random((len(h),1))).sum(1),joint.shape[1]-1)
    r=a%s.base.k
    y=s.base.execute_rule(h,q,r)
    return np.concatenate([h[:,1:],y[:,None]],1),r,read['read_hidden']


def dataset(split='train'):
    s=AdaptiveBeam(); rng=np.random.default_rng(9217)
    xx=[]; yy=[]; vv=[]; priors=[]
    for video in SPLITS[split]:
        full=load_video(video)
        y=full[:len(full)//2] if split=='train' else full[len(full)//2:]
        starts=np.linspace(63,len(y)-76,24,dtype=int)
        h=np.stack([y[t-31:t+1] for t in starts])
        q,m=s.machine.initialize(h); hidden=m['hidden']
        for drift in range(25):
            if drift in (0,8,24):
                pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
                pairs=[(i,int(e),int(r)) for i in range(len(h))
                       for e in s._top(pe[i],2) for r in s._top(tr[i,e],2)]
                ids=np.array([i for i,e,r in pairs]); rs=np.array([r for i,e,r in pairs])
                pred=s.base.execute_rule(h[ids],q[ids],rs)
                ch=np.concatenate([h[ids,1:],pred[:,None]],1)
                cq=rs; cm=read['read_hidden'][ids]
                xx.append(features(s.base,ch,cq,cm))
                costs=[]
                for t in range(1,51):
                    if t in (10,25,50):
                        truth=y[starts[ids]+drift+t]
                        a=np.c_[np.sin(ch[:,-1]),np.cos(ch[:,-1])]
                        b=np.c_[np.sin(truth),np.cos(truth)]
                        costs.append(((a-b)**2).mean(1))
                    if t<50: ch,cq,cm=step(s,ch,cq,cm)
                yy.append(np.mean(costs,axis=0))
                priors.extend([pe[i,e]*tr[i,e,r] for i,e,r in pairs])
                vv.extend([video]*len(pairs))
            if drift<24:h,q,hidden=step(s,h,q,hidden,rng)
    return np.concatenate(xx),np.concatenate(yy),np.array(vv),np.array(priors)


def main():
    torch.set_num_threads(1); torch.manual_seed(9217)
    x,y,video,prior=dataset()
    hold=np.isin(video,SPLITS['train'][-3:]); fit=~hold
    mean=x[fit].mean(0); scale=np.maximum(x[fit].std(0),1e-5)
    z=np.clip((x-mean)/scale,-8,8).astype('float32')
    model=torch.nn.Sequential(torch.nn.Linear(z.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    tx=torch.tensor(z[fit].reshape(-1,4,z.shape[1])); ty=torch.tensor(y[fit].reshape(-1,4))
    vx=torch.tensor(z[hold]); vy=y[hold].reshape(-1,4)
    diff=ty[:,:,None]-ty[:,None,:]
    mask=torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)[None] & (diff.abs()>1e-8)
    labels=(diff>0).float()
    best=None; best_cost=float('inf'); best_epoch=0
    for epoch in range(201):
        if epoch%10==0:
            with torch.no_grad():
                scores=model(vx).numpy().reshape(-1,4)
            cost=float(vy[np.arange(len(vy)),scores.argmin(1)].mean())
            if cost<best_cost:
                best_cost=cost;best_epoch=epoch
                best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch==200:break
        scores=model(tx).squeeze(-1)
        logits=scores[:,:,None]-scores[:,None,:]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[mask],labels[mask])
        optimizer.zero_grad();loss.backward();optimizer.step()
    model.load_state_dict(best)
    with torch.no_grad():scores=model(vx).numpy().reshape(-1,4)
    policy=prior[hold].reshape(-1,4).argmax(1)
    chosen=vy[np.arange(len(vy)),scores.argmin(1)]
    baseline=vy[np.arange(len(vy)),policy]
    report=dict(n_samples=len(x),drift_steps=[0,8,24],selected_epoch=best_epoch,
                training_videos=SPLITS['train'][:-3],validation_videos=SPLITS['train'][-3:],
                validation_selected_cost=float(chosen.mean()),policy_cost=float(baseline.mean()),
                mean_candidate_cost=float(vy.mean()),oracle_cost=float(vy.min(1).mean()),
                target='pairwise ranking of 10/25/50-step mean embedding MSE under fixed greedy continuation',
                note='Checkpoint selected on these held-out TRAIN videos; not an independent test.')
    group_videos=video[hold].reshape(-1,4)[:,0]
    report['per_video']={str(v):dict(ranker=float(chosen[group_videos==v].mean()),
                                   policy=float(baseline[group_videos==v].mean())) for v in SPLITS['train'][-3:]}
    out=Path('adaptive_search_results')
    np.savez(out/'ranker.npz',mean=mean,scale=scale,
             w1=best['0.weight'].numpy().T,b1=best['0.bias'].numpy(),
             w2=best['2.weight'].numpy().T,b2=best['2.bias'].numpy())
    (out/'ranker_training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def evaluate_dev():
    x,y,video,prior=dataset('dev')
    a=dict(np.load('adaptive_search_results/ranker.npz'))
    z=np.clip((x-a['mean'])/a['scale'],-8,8)
    scores=(np.tanh(z@a['w1']+a['b1'])@a['w2']+a['b2']).reshape(-1,4)
    costs=y.reshape(-1,4); ids=np.arange(len(costs))
    chosen=costs[ids,scores.argmin(1)]
    baseline=costs[ids,prior.reshape(-1,4).argmax(1)]
    videos=video.reshape(-1,4)[:,0]
    report=dict(n_samples=len(x),selected_cost=float(chosen.mean()),policy_cost=float(baseline.mean()),
                oracle_cost=float(costs.min(1).mean()),
                note='Frozen train-selected ranker on DEV-video tails; candidate ranking diagnostic, not search rollout RMSE.',
                per_video={str(v):dict(ranker=float(chosen[videos==v].mean()),
                                      policy=float(baseline[videos==v].mean())) for v in SPLITS['dev']})
    Path('adaptive_search_results/ranker_dev.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--evaluate-dev',action='store_true')
    if parser.parse_args().evaluate_dev:evaluate_dev()
    else:main()
