"""Exact small-model audit of pathwise plus routing-score writer gradients."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch


def paths(theta,detach_routing_state=False,return_history=False,return_marginal=False,truncate=False):
    actions=torch.tensor(list(itertools.product([0,1],repeat=4)),dtype=torch.long)
    x=torch.full((16,),.2,dtype=torch.float64);h=torch.full_like(x,.1);q=torch.zeros_like(x)
    logp=torch.zeros_like(x);history=[];prefix=[];marginal_logp=torch.zeros_like(x)
    for t in range(2):
        if truncate and t:x=x.detach();h=h.detach()
        e=actions[:,2*t].to(x.dtype);r=actions[:,2*t+1].to(x.dtype)
        rx=x.detach() if detach_routing_state else x
        rh=h.detach() if detach_routing_state else h
        read=.6*rh+.4*torch.tanh(rx+.3*q)
        pe=torch.sigmoid(.7*rx+.4*read-.2*q)
        pr=torch.sigmoid(-.5*rx+.8*read+.9*e+.3*q-.2)
        logp=logp+e*torch.log(pe)+(1-e)*torch.log1p(-pe)+r*torch.log(pr)+(1-r)*torch.log1p(-pr)
        pr0=torch.sigmoid(-.5*rx+.8*read+.3*q-.2)
        pr1=torch.sigmoid(-.5*rx+.8*read+.9+.3*q-.2)
        marginal=(1-pe)*pr0+pe*pr1
        marginal_logp=marginal_logp+r*torch.log(marginal)+(1-r)*torch.log1p(-marginal)
        # e influences r, not the numeric rule directly, matching the project interface.
        x=.85*x+.12*q-.2*r+.04*x*x+theta[0]+theta[1]*x
        h=read;q=r
        history.append(x);prefix.append(logp)
    if return_marginal:return x,logp,marginal_logp
    if return_history:return torch.stack(history,1),torch.stack(prefix,1)
    return x,logp


def energy_u(x,target):
    p=x.shape[1]
    if p<2:raise ValueError('at least two particles required')
    distance=torch.linalg.vector_norm(x-target,dim=-1).mean(1)
    pairs=torch.linalg.vector_norm(x[:,:,None]-x[:,None,:],dim=-1).sum((1,2))
    return distance-pairs/(2*p*(p-1))


def objectives(theta):
    x,logp=paths(theta);index=torch.tensor(list(itertools.product(range(16),repeat=3)))
    embeddings=torch.stack([torch.sin(x),torch.cos(x)],axis=-1)[index]
    target=torch.tensor([np.sin(.37),np.cos(.37)],dtype=torch.float64)
    cost=energy_u(embeddings,target);logs=logp[index];prob=logs.sum(1).exp()
    baselines=torch.stack([energy_u(embeddings[:,[j for j in range(3) if j!=i]],target) for i in range(3)],axis=1)
    pathwise=(prob.detach()*cost).sum()
    score=(prob.detach()*(cost.detach()[:,None]*logs).sum(1)).sum()
    score_loo=(prob.detach()*((cost[:,None]-baselines).detach()*logs).sum(1)).sum()
    return dict(exact=(prob*cost).sum(),pathwise=pathwise,score=score,
                hybrid=pathwise+score,hybrid_loo=pathwise+score_loo,
                probability_sum=prob.sum(),path_probability_sum=logp.exp().sum())


def audit(point):
    theta=torch.tensor(point,dtype=torch.float64,requires_grad=True);values=objectives(theta)
    gradients={k:torch.autograd.grad(values[k],theta,retain_graph=True)[0].detach().numpy()
               for k in ['exact','pathwise','score','hybrid','hybrid_loo']}
    finite=[]
    for dim in range(2):
        direction=np.zeros(2);direction[dim]=1e-5
        plus=objectives(torch.tensor(np.array(point)+direction,dtype=torch.float64))['exact']
        minus=objectives(torch.tensor(np.array(point)-direction,dtype=torch.float64))['exact']
        finite.append(float((plus-minus)/(2e-5)))
    np.testing.assert_allclose(gradients['hybrid'],gradients['exact'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(gradients['hybrid_loo'],gradients['exact'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(finite,gradients['exact'],rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(float(values['probability_sum'].detach()),1.,atol=1e-14)
    np.testing.assert_allclose(float(values['path_probability_sum'].detach()),1.,atol=1e-14)
    return dict(theta=list(point),objective=float(values['exact'].detach()),gradients={k:v.tolist() for k,v in gradients.items()},
        finite_difference=finite,pathwise_bias_norm=float(np.linalg.norm(gradients['pathwise']-gradients['exact'])),
        score_only_bias_norm=float(np.linalg.norm(gradients['score']-gradients['exact'])))


def main():
    torch.set_num_threads(1)
    results=[audit(p) for p in [(-.15,.07),(0.,0.),(.2,-.1)]]
    report=dict(results=results,paths=16,particles=3,joint_combinations=4096,
        note='Exact finite toy enumeration, not empirical task improvement. Full derivative through continuous state into later routing probabilities retained. score advantages/baselines detached; pathwise full ensemble loss remains differentiable. LOO baseline independent of omitted path conditional on theta. Real hard guards/discontinuities and NumPy/Torch parity still unaudited; no real-model training or DEV/TEST.')
    Path('adaptive_search_results/hybrid_writer_gradient.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
