"""Measure conditional rollout-noise variance of the initial actor gradient.

Same TRAIN histories, actor and objective; independent action RNG seeds.
No optimizer steps, no DEV/test data. This is not gradient variance from
changing training videos, nor a proof that more sampling solves generalization.
"""
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import Actor, prefix_windows, run_policy
from v20_rnn_mixture.engine.common import SPLITS


def main():
    torch.set_num_threads(1)
    s=AdaptiveBeam();windows=prefix_windows(SPLITS['train'][:-3],per_video=4)
    report=dict(n_batches=12,particles_per_window=4,n_windows=len(windows['history']),arms={})
    for name,feedback in [('control',False),('feedback',True)]:
        actor=Actor(s,feedback);grads=[];objectives=[]
        for i in range(12):
            actor.model.zero_grad(set_to_none=True)
            loss,kl,stats=run_policy(s,windows,actor=actor,training=True,seed=62000+i,particles=4)
            loss.backward()
            grads.append(np.concatenate([p.grad.detach().numpy().reshape(-1) for p in actor.model.parameters()]))
            objectives.append(stats['objective'])
        g=np.asarray(grads,dtype=float);norm=np.linalg.norm(g,axis=1)
        cosine=(g@g.T)/np.maximum(norm[:,None]*norm[None,:],1e-30)
        pairs=cosine[np.triu_indices(len(g),1)]
        mean=g.mean(0)
        noise_variance=((g-mean)**2).sum()/(len(g)-1)
        signal_sq=max(float(mean@mean-noise_variance/len(g)),0.)
        report['arms'][name]=dict(mean_gradient_norm=float(np.linalg.norm(mean)),
            batch_gradient_norm_mean=float(norm.mean()),
            pairwise_cosine_mean=float(pairs.mean()),pairwise_cosine_median=float(np.median(pairs)),
            negative_cosine_fraction=float((pairs<0).mean()),
            estimated_gradient_noise_trace=float(noise_variance),
            noise_corrected_signal_squared=signal_sq,
            estimated_single_batch_signal_to_noise=float(np.sqrt(signal_sq/max(noise_variance,1e-30))),
            objective_mean=float(np.mean(objectives)),objective_std=float(np.std(objectives)),
            note='Finite12-batch estimate; dependent gradient-pair cosines are descriptive, not independent observations.')
        print(name,report['arms'][name],flush=True)
    Path('adaptive_search_results/policy_gradient_noise.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
