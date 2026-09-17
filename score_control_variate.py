"""Scalar zero-mean score control: coefficient must be fit independently."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from hybrid_writer_gradient import paths,energy_u,objectives


def fit_coefficient(gradient,score,weights=None):
    g=np.asarray(gradient,dtype=float);s=np.asarray(score,dtype=float)
    if g.shape!=s.shape or g.ndim!=2 or not np.isfinite(g).all() or not np.isfinite(s).all():raise ValueError('Finite matching [samples,parameters] required')
    w=np.ones(len(g))/len(g) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(len(g),) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Nonnegative nonempty weights required')
    w=w/w.sum();den=float(np.sum(w*np.sum(s*s,axis=1)))
    return 0. if den<1e-30 else float(np.sum(w*np.sum(g*s,axis=1))/den)


def toy_samples(theta):
    x,lp=paths(theta);idx=torch.tensor(list(itertools.product(range(16),repeat=3)))
    emb=torch.stack([torch.sin(x),torch.cos(x)],-1)[idx]
    target=torch.tensor([np.sin(.37),np.cos(.37)],dtype=torch.float64)
    cost=energy_u(emb,target);base=torch.stack([energy_u(emb[:,[j for j in range(3) if j!=i]],target) for i in range(3)],1)
    logs=lp[idx];surrogate=cost+((cost[:,None]-base).detach()*logs).sum(1)
    return surrogate,logs.sum(1)


def audit(point):
    theta=torch.tensor(point,dtype=torch.float64,requires_grad=True)
    g,s=torch.func.jacfwd(toy_samples)(theta);g=g.detach().numpy();s=s.detach().numpy()
    _,logs=toy_samples(theta);w=logs.detach().exp().numpy()
    mean=(w[:,None]*g).sum(0);score_mean=(w[:,None]*s).sum(0)
    exact=torch.autograd.grad(objectives(theta)['exact'],theta)[0].detach().numpy()
    np.testing.assert_allclose(mean,exact,rtol=1e-11,atol=1e-12);np.testing.assert_allclose(score_mean,0,atol=1e-14)
    coefficient=fit_coefficient(g,s,w);adjusted=g-coefficient*s
    corrected_mean=(w[:,None]*adjusted).sum(0)
    np.testing.assert_allclose(corrected_mean,mean,rtol=1e-11,atol=1e-12)
    variance=lambda z:float(np.sum(w*np.sum((z-(w[:,None]*z).sum(0))**2,axis=1)))
    before=variance(g);after=variance(adjusted);assert after<=before+1e-12
    # Fit one coefficient from the very evaluation sample: finite, but biased.
    sample_fit=np.sum(g*s,axis=1)/(np.sum(s*s,axis=1)+1e-4)
    leaked_mean=(w[:,None]*(g-sample_fit[:,None]*s)).sum(0)
    # Independent fit/evaluation draws factorize; this is exact, not an MC estimate.
    independent_mean=mean-float(np.dot(w,sample_fit))*score_mean
    np.testing.assert_allclose(independent_mean,mean,atol=1e-12)
    bias=float(np.linalg.norm(leaked_mean-mean));assert bias>1e-8
    return dict(theta=list(point),coefficient=coefficient,mean=mean.tolist(),corrected_mean=corrected_mean.tolist(),score_mean=score_mean.tolist(),
                variance_before=before,variance_after=after,same_sample_fit_bias_norm=bias,independent_fit_mean=independent_mean.tolist())


def main():
    torch.set_num_threads(1);results=[audit(p) for p in [(-.15,.07),(0.,0.),(.2,-.1)]]
    report=dict(results=results,note='Exact4096jointpaths/P3 full differentiable toy. Fixed/population-optimal coefficient preserves mean; independent random fitted coefficient also does by factorization. Same-sample ridge fit is biased. No claim real-task variance decreases or truncated estimator becomes unbiased for original objective.')
    Path('adaptive_search_results/score_control_variate_toy.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
