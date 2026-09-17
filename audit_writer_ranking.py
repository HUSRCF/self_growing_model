"""Fixed TRAIN candidate sets: action-RNG ranking reproducibility."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_writer import constant_model


def ranking_summary(original,fresh):
    original=np.asarray(original);fresh=np.asarray(fresh)
    chosen=int(np.argmin(original));winners=np.argmin(fresh,axis=1)
    delta=fresh[:,chosen]-fresh[:,0]
    return dict(original_chosen=chosen,original_difference=float(original[chosen]-original[0]),
                fresh_winners=winners.tolist(),same_winner_fraction=float(np.mean(winners==chosen)),
                mean_costs=fresh.mean(0).tolist(),mean_winner=int(np.argmin(fresh.mean(0))),
                chosen_mean_difference=float(delta.mean()),
                chosen_conditional_rng_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                chosen_better_replicas=int((delta<0).sum()),
                group3_winners=[int(np.argmin(fresh[i:i+3].mean(0))) for i in [0,3]],
                group3_cross_evaluation=[float((fresh[3:,int(np.argmin(fresh[:3].mean(0)))]-fresh[3:,0]).mean()),
                                         float((fresh[:3,int(np.argmin(fresh[3:].mean(0)))]-fresh[:3,0]).mean())])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    root=Path('adaptive_search_results');path=root/f'closed_loop_writer_seed{args.seed}.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest();source=json.loads(path.read_text())
    s=AdaptiveBeam();pool=TrainingPrefixPool(s.base,steps=300);sets=[]
    for iteration in [1,6,12]:
        row=source['trace'][iteration-1];w=pool.sample(row['window_seed'],per_video=2)
        np.testing.assert_array_equal(w['video'],row['video']);np.testing.assert_array_equal(w['start'],row['start'])
        models=[constant_model(np.array(theta),np.array(source['bound'])) for theta in row['candidates']]
        original=[rollout(s,w,m,row['action_seed'],particles=4)[0]['objective'] for m in models]
        np.testing.assert_array_equal(original,row['costs'])
        costs=[]
        for action_seed in range(81017,81023):
            assert action_seed!=row['action_seed']
            costs.append([rollout(s,w,m,action_seed,particles=4)[0]['objective'] for m in models])
        summary=ranking_summary(original,costs)
        sets.append(dict(iteration=iteration,original=row,fresh_seeds=list(range(81017,81023)),
                         fresh_costs=costs,summary=summary))
        print(args.seed,iteration,summary,flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(optimization_seed=args.seed,sets=sets,source_sha256=digest,source_unchanged=True,
                exact_original_cost_replay=True,
                note='Prespecified iterations1/6/12 from each of3optimization seeds. Same fixed fit-TRAIN windows/candidates,6new actionseeds,P4/300. No model updates or hold/DEV/TEST. Candidates selected on original RNG; new mean only diagnostic. Group3 comparison descriptive, small correlated set count, no capacity/optimization impossibility claim.')
    (root/f'writer_ranking_audit_seed{args.seed}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
