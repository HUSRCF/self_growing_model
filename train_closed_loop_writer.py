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


def checkpoint_iterations(iterations):
    if iterations==12:return {4,8,12}
    if iterations==4:return {1,2,4}
    raise ValueError('Predeclared schedules are 4 or 12 updates')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=1901)
    parser.add_argument('--particles',type=int,choices=[4,12],default=4)
    parser.add_argument('--iterations',type=int,choices=[4,12],default=12)
    parser.add_argument('--replay',action='store_true');args=parser.parse_args()
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
    for iteration in range(1,args.iterations+1):
        window_seed=args.seed*100+iteration;action_seed=args.seed*1000+iteration
        w=pool.sample(window_seed,per_video=2);proposals=candidates(theta,rng.normal(size=2),iteration)
        costs=[rollout(s,w,constant_model(p,bound),action_seed,particles=args.particles)[0]['objective'] for p in proposals]
        chosen=int(np.argmin(costs));theta=proposals[chosen].copy()
        trace.append(dict(iteration=iteration,window_seed=window_seed,action_seed=action_seed,
                          video=w['video'].tolist(),start=w['start'].tolist(),
                          candidates=proposals.tolist(),costs=costs,chosen=chosen))
        print(args.seed,'train',iteration,chosen,costs,flush=True)
        if iteration in checkpoint_iterations(args.iterations):evaluate_checkpoint(iteration)
    selected=min(checkpoints,key=lambda c:c['objective'])
    model=constant_model(np.array(selected['theta']),bound)
    stem=f'closed_loop_writer_seed{args.seed}' if (args.particles,args.iterations)==(4,12) else f'closed_loop_writer_p{args.particles}_i{args.iterations}_seed{args.seed}'
    report=dict(seed=args.seed,bound=bound.tolist(),trace=trace,checkpoints=checkpoints,
                selected_epoch=selected['epoch'],selected_theta=selected['theta'],
                selected_objective=selected['objective'],sampler=pool.audit(),
                training_rollouts=3*args.iterations,training_particle_transitions=3*args.iterations*20*args.particles*300,
                config=dict(particles=args.particles,iterations=args.iterations),
                note='3full candidates per update,current+antithetic direction,shared RNG and refreshed fit windows. No gradients through discrete sampling; noisy direct search, no unbiased-gradient claim.2constant parameters,zero-init,staticfit-onlybound. TRAIN10fit/3reusedhold,checkpoint0eligible,4holdchecks x2seeds. Fewer updates change coverage and step-radius sequence. No DEV/TEST/checker enforcement/default promotion.')
    if args.replay:
        original=json.loads((root/f'{stem}.json').read_text())
        for key in ['trace','checkpoints','selected_epoch','selected_theta','selected_objective','sampler']:
            assert report[key]==original[key],key
        with np.load(root/f'{stem}.npz') as saved:
            for key,value in model.items():np.testing.assert_array_equal(value,saved[key])
        report['previous_exact_replay']=True;stem+='_replay'
    np.savez(root/f'{stem}.npz',**model)
    (root/f'{stem}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
