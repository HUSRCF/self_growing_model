"""Fixed-parameter, fixed-window truncated gradient direction reproducibility."""
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from train_hybrid_writer import hybrid_loss
from training_window_sampler import TrainingPrefixPool


def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(np.dot(a,b)/den)


def stability(samples):
    x=np.asarray(samples,dtype=float).reshape(len(samples),-1)
    if len(x)<2:raise ValueError('At least two independent RNG repetitions required')
    mean=x.mean(0);trace=float(np.sum(x.var(0,ddof=1)));sem=np.sqrt(trace/len(x))
    pairs=[cosine(x[i],x[j]) for i in range(len(x)) for j in range(i)]
    valid=[p for p in pairs if p is not None]
    loo=[cosine(x[i],np.delete(x,i,axis=0).mean(0)) for i in range(len(x))]
    return dict(mean=mean.tolist(),variance_trace=trace,mean_norm=float(np.linalg.norm(mean)),
                rms_standard_error=float(sem),mean_to_rms_se=None if sem==0 else float(np.linalg.norm(mean)/sem),
                pairwise_cosine_mean=None if not valid else float(np.mean(valid)),
                negative_pair_count=sum(p<0 for p in valid),valid_pairs=len(valid),
                leave_one_seed_out_cosines=loo,first_half_vs_second_half_cosine=cosine(x[:len(x)//2].mean(0),x[len(x)//2:].mean(0)),
                positive_component_counts=(x>0).sum(0).tolist())


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s);root=Path('adaptive_search_results')
    w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1)
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    rows=[]
    for seed in range(371017,371025):
        total=np.zeros((8,2));path=np.zeros((8,2));visits=np.zeros(8,dtype=int);guards=0;windows=[]
        for j,h in enumerate(w['history']):
            initial=tensor(np.repeat(h[None],4,axis=0));theta=torch.zeros((8,2),dtype=torch.float64,requires_grad=True)
            with torch.no_grad():q0,_=b.initialize(initial)
            a=sampled_path(b,initial,300,theta*bound,seed+j*100000,trace=True,detach_every=50)
            source=torch.stack([q0]+[t['q'] for t in a['trace'][:-1]]).numpy()
            counts=np.bincount(source.reshape(-1),minlength=8);visits+=counts
            guards+=int(a['failed'].any(1).sum())
            cost,loss=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'])
            pg=torch.autograd.grad(cost,theta,retain_graph=True)[0].detach().numpy()
            g=torch.autograd.grad(loss,theta)[0].detach().numpy()
            assert np.isfinite(g).all() and np.isfinite(pg).all()
            total+=g/len(w['history']);path+=pg/len(w['history'])
            windows.append(dict(video=int(w['video'][j]),start=int(w['start'][j]),gradient=g.tolist(),path_gradient=pg.tolist(),source_visits=counts.tolist()))
            del a,cost,loss
        row=dict(seed=seed,gradient=total.tolist(),path_gradient=path.tolist(),source_visits=visits.tolist(),guard_particles=guards,windows=windows)
        assert visits.sum()==10*4*300;rows.append(row);print(seed,'shared',total.sum(0).tolist(),'qnorm',float(np.linalg.norm(total)),flush=True)
    g=np.array([r['gradient'] for r in rows]);pg=np.array([r['path_gradient'] for r in rows])
    summary={'q_table':stability(g),'shared_sum':stability(g.sum(1)),'path_only_table':stability(pg),
             'per_q':{str(q):dict(stability(g[:,q]),visits=[r['source_visits'][q] for r in rows]) for q in range(8)}}
    report=dict(rows=rows,summary=summary,video=w['video'].tolist(),window_start=w['start'].tolist(),
                note='theta0,fixed10TRAIN-fit windows seed190101,P4/300,biased50step detach,8new action seeds371017–24 with window offset j*100000. Shared gradient obtained by exact sum_q chain rule at tied zero table,previously verified against directshared. Visits are correlated particle-steps,not independent samples. Mean/SE/cosines descriptive at fixed states,not original-objective unbiasedness proof. No training/hold/DEV/TEST.')
    (root/'source_q_gradient_audit.json').write_text(json.dumps(report,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
