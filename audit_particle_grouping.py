"""Pooled versus grouped U-energy, exactly the same particle trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from ensemble_score_objective import energy_costs
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_writer import constant_model


def pooled_grouped_cost(x,y,failed,groups):
    p=x.shape[1];indices=np.concatenate(groups)
    if not np.array_equal(np.sort(indices),np.arange(p)):raise ValueError('Groups must partition all particles')
    if len({len(g) for g in groups})!=1:raise ValueError('Equal groups required')
    pooled=energy_costs(x,y,failed)[0]
    grouped=np.mean([energy_costs(x[:,g],y,failed[:,g])[0] for g in groups],axis=0)
    return pooled,grouped


def trajectory_costs(pred,truth,failed):
    values=[]
    for t in [50,100,300]:
        x=pred[:,:,t-1];y=truth[:,t-1]
        emb=np.concatenate([np.sin(x),np.cos(x)],-1);target=np.c_[np.sin(y),np.cos(y)]
        values.append(pooled_grouped_cost(emb,target,failed[:,:,t-1],np.arange(12).reshape(3,4)))
    return np.array(values).mean((0,2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    root=Path('adaptive_search_results');path=root/f'closed_loop_writer_seed{args.seed}.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest();source=json.loads(path.read_text())
    s=AdaptiveBeam();pool=TrainingPrefixPool(s.base,steps=300);sets=[]
    for iteration in [1,6,12]:
        row=source['trace'][iteration-1];w=pool.sample(row['window_seed'],per_video=2)
        np.testing.assert_array_equal(w['video'],row['video']);np.testing.assert_array_equal(w['start'],row['start'])
        costs=[]
        for seed in range(91017,91023):
            values=[]
            for theta in row['candidates']:
                result,pred,failed=rollout(s,w,constant_model(np.array(theta),np.array(source['bound'])),seed,particles=12)
                value=trajectory_costs(pred,w['truth'],failed)
                np.testing.assert_allclose(value[0],result['objective'],rtol=1e-14,atol=1e-14)
                values.append(value.tolist())
            costs.append(values)
        a=np.array(costs);summary={}
        for method,column in [('pooled12',0),('mean3x4',1)]:
            c=a[:,:,column];diff=c[:,1:]-c[:,:1]
            summary[method]=dict(costs=c.tolist(),mean_costs=c.mean(0).tolist(),
                paired_mean=diff.mean(0).tolist(),paired_variance=diff.var(0,ddof=1).tolist(),
                winners=np.argmin(c,axis=1).tolist(),half_mean_winners=[int(np.argmin(c[i:i+3].mean(0))) for i in [0,3]])
        sets.append(dict(iteration=iteration,summary=summary,action_seeds=list(range(91017,91023))))
        print(args.seed,iteration,{k:v['paired_variance'] for k,v in summary.items()},flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(optimization_seed=args.seed,sets=sets,source_sha256=digest,source_unchanged=True,
                same_particle_trajectories=True,pooled_matches_rollout=True,
                note='Same12trajectories pooled vs3disjoint4-particle U-statistic averages. Same particle transitions, not measured equalCPU implementations. FixedTRAINfit9candidate sets/6newactionseeds, no fit/hold/DEV/TEST. Candidate-difference variance has only6replicates, descriptive not independent-video inference.')
    (root/f'particle_grouping_seed{args.seed}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
