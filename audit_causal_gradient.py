"""Exact toy expectation and fixed-state paired real gradient variability."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from hybrid_writer_gradient import paths,energy_u
from train_hybrid_writer import hybrid_loss
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool


def toy(point):
    theta=torch.tensor(point,dtype=torch.float64,requires_grad=True)
    x,lp=paths(theta,return_history=True);idx=torch.tensor(list(itertools.product(range(16),repeat=3)))
    prob=lp[:,-1][idx].sum(1).exp();costs=[];full=[];causal=[]
    for t in range(2):
        emb=torch.stack([torch.sin(x[:,t]),torch.cos(x[:,t])],-1)[idx]
        target=tensor([np.sin(.37),np.cos(.37)]);cost=energy_u(emb,target)
        baseline=torch.stack([energy_u(emb[:,[j for j in range(3) if j!=i]],target) for i in range(3)],1)
        adv=(cost[:,None]-baseline).detach();costs.append(cost)
        full.append((adv*lp[:,-1][idx]).sum(1));causal.append((adv*lp[:,t][idx]).sum(1))
    cost=torch.stack(costs).mean(0)
    values={'exact':(prob*cost).sum(),'full':(prob.detach()*(cost+torch.stack(full).mean(0))).sum(),
            'causal':(prob.detach()*(cost+torch.stack(causal).mean(0))).sum()}
    g={k:torch.autograd.grad(v,theta,retain_graph=True)[0].detach().numpy() for k,v in values.items()}
    for k in ['full','causal']:np.testing.assert_allclose(g[k],g['exact'],rtol=1e-11,atol=1e-12)
    return dict(theta=list(point),gradients={k:v.tolist() for k,v in g.items()})


def main():
    torch.set_num_threads(1);proof=[toy(p) for p in [(-.15,.07),(0.,0.),(.2,-.1)]]
    s=AdaptiveBeam();b=FrozenBridge(s);pool=TrainingPrefixPool(s.base,steps=300)
    w=pool.sample(190101,per_video=1);root=Path('adaptive_search_results')
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    rows=[]
    for seed in range(321017,321025):
        grads={k:np.zeros(2) for k in ['path','full','causal']};guard=0
        for j,h in enumerate(w['history']):
            theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
            a=sampled_path(b,tensor(np.repeat(h[None],4,axis=0)),300,theta*bound,seed+j*100000)
            cost,full=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'])
            _,causal=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'],a['prefix_logp'])
            for k,v in [('path',cost),('full',full),('causal',causal)]:
                g=torch.autograd.grad(v,theta,retain_graph=k!='causal')[0].detach().numpy()
                assert np.isfinite(g).all();grads[k]+=g/len(w['history'])
            guard+=int(a['failed'].any(1).sum());del a,cost,full,causal
        rows.append(dict(seed=seed,gradients={k:v.tolist() for k,v in grads.items()},guard_particles=guard));print(rows[-1],flush=True)
    summary={}
    for k in ['path','full','causal']:
        g=np.array([r['gradients'][k] for r in rows]);cov=np.cov(g,rowvar=False)
        summary[k]=dict(mean=g.mean(0).tolist(),covariance=cov.tolist(),variance_trace=float(np.trace(cov)))
    report=dict(toy=proof,rows=rows,summary=summary,video=w['video'].tolist(),window_start=w['start'].tolist(),
                note='theta0,fixed10fitTRAIN-prefix windows,P4,8paired action seeds. Raw gradients, no clipping/training/selection/DEV/TEST. Path-only is biased control. Empirical covariance at fixed states, small8seed sample; no claimed universal variance reduction or hard-guard unbiasedness.')
    (root/'causal_gradient_audit.json').write_text(json.dumps(report,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
