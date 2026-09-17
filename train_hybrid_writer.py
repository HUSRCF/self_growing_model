"""Small bounded TRAIN pilot of pathwise plus routing-score gradients."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from hybrid_writer_gradient import energy_u
from ensemble_score_objective import energy_costs
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from train_closed_loop_writer import constant_model
from v20_rnn_mixture.engine.common import SPLITS


def hybrid_loss(prediction,truth,failed,logp):
    """One window, independent particles; full-horizon score with detached LOO."""
    costs=[];baselines=[]
    for step in [50,100,300]:
        x=prediction[:,step-1];y=truth[step-1]
        emb=torch.cat([torch.sin(x),torch.cos(x)],-1)[None]
        target=torch.cat([torch.sin(y),torch.cos(y)],-1)
        costs.append(energy_u(emb,target)[0]+2*failed[:,step-1].to(x.dtype).mean())
        _,baseline=energy_costs(emb.detach().numpy(),target.detach().numpy()[None],failed[:,step-1].numpy()[None])
        baselines.append(tensor(baseline[0]))
    cost=torch.stack(costs).mean();baseline=torch.stack(baselines).mean(0)
    score=((cost.detach()-baseline)*logp).sum()
    return cost,cost+score


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s);pool=TrainingPrefixPool(s.base,steps=300)
    root=Path('adaptive_search_results')
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as saved:bound=saved['cap'].copy()*.01
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);allruns=[]
    for seed in [1901,2718,3141]:
        theta=torch.zeros(2,dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([theta],lr=.1)
        checkpoints=[];trace=[]
        def evaluate(epoch):
            model=constant_model(theta.detach().numpy(),bound)
            rows=[rollout(s,hold,model,a)[0] for a in [301017,301018]]
            checkpoints.append(dict(epoch=epoch,theta=theta.detach().tolist(),objective=float(np.mean([r['objective'] for r in rows])),runs=rows))
            print(seed,'hold',epoch,checkpoints[-1]['objective'],flush=True)
        evaluate(0)
        for iteration in range(1,5):
            start=time.perf_counter();w=pool.sample(seed*100+iteration,per_video=1)
            total=torch.zeros_like(theta);path=torch.zeros_like(theta);costs=[];guards=0;finite=True
            for j,h in enumerate(w['history']):
                a=sampled_path(b,tensor(np.repeat(h[None],4,axis=0)),300,theta*tensor(bound),seed*10000+iteration*100+j)
                guards+=int(a['failed'].any(1).sum())
                cost,loss=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'])
                pg=torch.autograd.grad(cost,theta,retain_graph=True)[0]
                gradient=torch.autograd.grad(loss,theta)[0]
                finite=finite and bool(torch.isfinite(gradient).all() and torch.isfinite(pg).all() and torch.isfinite(cost))
                total+=gradient.detach()/len(w['history']);path+=pg.detach()/len(w['history']);costs.append(float(cost.detach()))
                del a,cost,loss,pg,gradient
            skipped=guards>0 or not finite
            before=theta.detach().tolist()
            if not skipped:
                opt.zero_grad();theta.grad=total.clone();torch.nn.utils.clip_grad_norm_([theta],1.);opt.step()
                with torch.no_grad():theta.clamp_(-1,1)
            row=dict(iteration=iteration,before=before,after=theta.detach().tolist(),objective=float(np.mean(costs)),
                     path_gradient=path.tolist(),hybrid_gradient=total.tolist(),guards=guards,finite=finite,skipped=skipped,
                     seconds=time.perf_counter()-start,video=w['video'].tolist(),window_start=w['start'].tolist())
            trace.append(row);print(seed,'train',iteration,row,flush=True)
            if iteration in [1,2,4]:evaluate(iteration)
        selected=min(checkpoints,key=lambda c:c['objective'])
        run=dict(seed=seed,trace=trace,checkpoints=checkpoints,selected_epoch=selected['epoch'],selected_theta=selected['theta'])
        allruns.append(run)
        report=dict(bound=bound.tolist(),runs=allruns,config=dict(iterations=4,particles=4,windows_per_update=10,steps=300,lr=.1,clip=1.),
                    note='TRAIN only; original NumPy checkpoint evaluator. No DEV/TEST/default promotion. Full pathwise plus detached leave-one-particle routing score, no truncated BPTT. Whole update skipped on any guard or nonfinite gradient. Hard-boundary gradient not claimed unbiased. Three optimization seeds retained; budget not matched to prior direct search.')
        (root/'hybrid_writer_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
