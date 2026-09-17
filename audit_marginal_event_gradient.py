"""Integrate nuisance event labels in routing score, preserving sampled paths."""
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
    x,lp,lm=paths(theta,return_marginal=True)
    joint=torch.autograd.functional.jacobian(lambda t:paths(t,return_marginal=True)[1],theta,vectorize=True)
    marginal=torch.autograd.functional.jacobian(lambda t:paths(t,return_marginal=True)[2],theta,vectorize=True)
    acts=torch.tensor(list(itertools.product([0,1],repeat=4)));groups=2*acts[:,1]+acts[:,3];errors=[]
    for k in range(4):
        mask=groups==k;weights=lp[mask].detach().softmax(0)
        conditional=(weights[:,None]*joint[mask]).sum(0)
        expected=marginal[mask][0]
        np.testing.assert_allclose(conditional,expected,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(lp[mask].exp().sum().detach(),lm[mask][0].exp().detach(),atol=1e-14)
        errors.append(float((conditional-expected).abs().max()))
    idx=torch.tensor(list(itertools.product(range(16),repeat=3)))
    emb=torch.stack([torch.sin(x),torch.cos(x)],-1)[idx];target=tensor([np.sin(.37),np.cos(.37)])
    cost=energy_u(emb,target);baseline=torch.stack([energy_u(emb[:,[j for j in range(3) if j!=i]],target) for i in range(3)],1)
    prob=lp[idx].sum(1).exp();adv=(cost[:,None]-baseline).detach()
    values={'exact':(prob*cost).sum(),'joint':(prob.detach()*(cost+(adv*lp[idx]).sum(1))).sum(),
            'marginal':(prob.detach()*(cost+(adv*lm[idx]).sum(1))).sum()}
    gradients={k:torch.autograd.grad(v,theta,retain_graph=True)[0].detach().numpy() for k,v in values.items()}
    for k in ['joint','marginal']:np.testing.assert_allclose(gradients[k],gradients['exact'],rtol=1e-11,atol=1e-12)
    return dict(theta=list(point),conditional_score_max_error=max(errors),gradients={k:v.tolist() for k,v in gradients.items()})


def main():
    torch.set_num_threads(1);proof=[toy(p) for p in [(-.15,.07),(0.,0.),(.2,-.1)]]
    root=Path('adaptive_search_results');old=json.loads((root/'causal_gradient_audit.json').read_text())
    s=AdaptiveBeam();b=FrozenBridge(s);w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1)
    np.testing.assert_array_equal(w['start'],old['window_start'])
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    rows=[]
    for oldrow in old['rows']:
        seed=oldrow['seed'];grads={k:np.zeros(2) for k in ['joint','marginal']};windows=[];guards=0
        for j,h in enumerate(w['history']):
            theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
            a=sampled_path(b,tensor(np.repeat(h[None],4,axis=0)),300,theta*bound,seed+j*100000)
            pair={}
            for k,logp in [('joint',a['logp']),('marginal',a['marginal_logp'])]:
                _,loss=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],logp)
                g=torch.autograd.grad(loss,theta,retain_graph=k=='joint')[0].detach().numpy()
                assert np.isfinite(g).all();grads[k]+=g/len(w['history']);pair[k]=g.tolist()
            guards+=int(a['failed'].any(1).sum());windows.append(pair);del a,loss
        np.testing.assert_array_equal(grads['joint'],oldrow['gradients']['full'])
        row=dict(seed=seed,gradients={k:v.tolist() for k,v in grads.items()},windows=windows,guard_particles=guards)
        rows.append(row);print(seed,row['gradients'],flush=True)
    summary={}
    for k in ['joint','marginal']:
        g=np.array([r['gradients'][k] for r in rows]);summary[k]=dict(mean=g.mean(0).tolist(),variance_trace=float(np.trace(np.cov(g,rowvar=False))))
    report=dict(toy=proof,rows=rows,summary=summary,old_joint_gradients_exact=True,
                note='Same frozen theta0/10fit windows/P4/300/8action seeds as causal audit; same two uniforms and same trajectory per paired estimator. No training/clipping/DEV/TEST. Event may be marginalized only because current F and read-hidden have no direct dependence on sampled event. Toy conditional-score identity does not resolve hard-boundary gradients.')
    (root/'marginal_event_gradient_audit.json').write_text(json.dumps(report,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
