"""Low-dimensional bounded direct search of actual closed-loop writer cost."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def candidates(theta,direction,iteration):
    d=np.asarray(direction,dtype=float);d=d/max(np.linalg.norm(d),1e-30)
    radius=.5/np.sqrt(iteration)
    return np.stack([theta,np.clip(theta+radius*d,-1,1),np.clip(theta-radius*d,-1,1)])


def constant_model(theta,bound):
    if np.any(np.abs(theta)>1):raise ValueError('Parameter outside fixed box')
    return dict(kind='constant',value=np.asarray(theta)*bound)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=1901);args=parser.parse_args()
    root=Path('adaptive_search_results');s=AdaptiveBeam();pool=TrainingPrefixPool(s.base,steps=300)
    # Frozen bound was fitted only on the ten fitting TRAIN videos.
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as saved:bound=saved['cap'].copy()*.01
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    rng=np.random.default_rng(args.seed);theta=np.zeros(2);trace=[];checkpoints=[]
    def evaluate_checkpoint(epoch):
        rows=[rollout(s,hold,constant_model(theta,bound),seed)[0] for seed in [62017,62019]]
        point=dict(epoch=epoch,theta=theta.tolist(),value=(theta*bound).tolist(),runs=rows,
                   objective=float(np.mean([r['objective'] for r in rows])))
        checkpoints.append(point);print(args.seed,'hold',epoch,point['objective'],flush=True)
    evaluate_checkpoint(0)
    for iteration in range(1,13):
        window_seed=args.seed*100+iteration;action_seed=args.seed*1000+iteration
        w=pool.sample(window_seed,per_video=2);proposals=candidates(theta,rng.normal(size=2),iteration)
        costs=[rollout(s,w,constant_model(p,bound),action_seed,particles=4)[0]['objective'] for p in proposals]
        chosen=int(np.argmin(costs));theta=proposals[chosen].copy()
        trace.append(dict(iteration=iteration,window_seed=window_seed,action_seed=action_seed,
                          video=w['video'].tolist(),start=w['start'].tolist(),
                          candidates=proposals.tolist(),costs=costs,chosen=chosen))
        print(args.seed,'train',iteration,chosen,costs,flush=True)
        if iteration%4==0:evaluate_checkpoint(iteration)
    selected=min(checkpoints,key=lambda c:c['objective'])
    model=constant_model(np.array(selected['theta']),bound)
    np.savez(root/f'closed_loop_writer_seed{args.seed}.npz',**model)
    report=dict(seed=args.seed,bound=bound.tolist(),trace=trace,checkpoints=checkpoints,
                selected_epoch=selected['epoch'],selected_theta=selected['theta'],
                selected_objective=selected['objective'],sampler=pool.audit(),
                training_rollouts=36,training_particle_transitions=36*20*4*300,
                note='12iterations,3full candidates each,current+antithetic direction,shared RNG and refreshed fit windows. No gradients through discrete sampling; noisy direct search, no unbiased-gradient claim.2constant parameters,zero-init,staticfit-onlybound. TRAIN10fit/3reusedhold,checkpoint0eligible,4holdchecks x2seeds. No DEV/TEST/checker enforcement/default promotion.')
    (root/f'closed_loop_writer_seed{args.seed}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
