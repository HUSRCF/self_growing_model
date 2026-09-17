"""Check unchanged forward values and explicitly exhibit truncation bias."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from hybrid_writer_gradient import paths,energy_u,objectives
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from training_window_sampler import TrainingPrefixPool
from train_hybrid_writer import hybrid_loss


def toy(point):
    theta=torch.tensor(point,dtype=torch.float64,requires_grad=True)
    x,lp=paths(theta);xx,ll=paths(theta,truncate=True)
    np.testing.assert_array_equal(x.detach(),xx.detach());np.testing.assert_array_equal(lp.detach(),ll.detach())
    idx=torch.tensor(list(itertools.product(range(16),repeat=3)))
    emb=torch.stack([torch.sin(xx),torch.cos(xx)],-1)[idx];target=tensor([np.sin(.37),np.cos(.37)])
    cost=energy_u(emb,target);baseline=torch.stack([energy_u(emb[:,[j for j in range(3) if j!=i]],target) for i in range(3)],1)
    surrogate=(lp[idx].sum(1).exp().detach()*(cost+((cost[:,None]-baseline).detach()*ll[idx]).sum(1))).sum()
    truncated=torch.autograd.grad(surrogate,theta)[0].detach().numpy()
    exact=torch.autograd.grad(objectives(theta)['exact'],theta)[0].detach().numpy()
    assert np.linalg.norm(truncated-exact)>1e-8
    return dict(theta=list(point),exact=exact.tolist(),truncated=truncated.tolist(),bias_norm=float(np.linalg.norm(truncated-exact)))


def main():
    torch.set_num_threads(1);proof=[toy(p) for p in [(-.15,.07),(0.,0.),(.2,-.1)]]
    s=AdaptiveBeam();b=FrozenBridge(s);w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1);root=Path('adaptive_search_results')
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    old=json.loads((root/'marginal_event_gradient_audit.json').read_text());rows=[]
    for seed,j in [(321017,0),(321020,9),(321023,7)]:
        variants={};gradients={}
        for length in [0,50]:
            theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
            a=sampled_path(b,tensor(np.repeat(w['history'][j:j+1],4,axis=0)),300,theta*bound,seed+j*100000,detach_every=length)
            cost,loss=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'])
            g=torch.autograd.grad(loss,theta)[0].detach().numpy();assert np.isfinite(g).all()
            gradients[str(length)]=g.tolist();variants[length]={k:a[k].detach().numpy() for k in ['prediction','failed','logp','history','q','hidden']}
            del a,cost,loss
        for key in variants[0]:np.testing.assert_array_equal(variants[0][key],variants[50][key])
        previous=next(r for r in old['rows'] if r['seed']==seed)['windows'][j]['joint']
        np.testing.assert_allclose(gradients['0'],previous,rtol=1e-12,atol=1e-9)
        rows.append(dict(seed=seed,window_index=j,gradients=gradients,forward_exact=True));print(rows[-1],flush=True)
    (root/'truncated_gradient_audit.json').write_text(json.dumps(dict(toy=proof,rows=rows,note='Forward exact, gradients intentionally biased; no benefit claimed from smaller norm. No training/DEV/TEST.'),indent=2))


if __name__=='__main__':main()
