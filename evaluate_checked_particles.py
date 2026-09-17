"""DEV compute/quality frontier for the unchanged batched checked runtime."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from evaluate_adapted_checked import verify_checks
from stateless_policy_adapter import StatelessPolicyAdapter
from v20_rnn_mixture.engine.runtime import CheckedMachine
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.evaluate import metrics
from v20_rnn_mixture.engine.common import DT


def horizon_metrics(pred,truth,failed,horizons=(10,25,50,100,300)):
    # Energy is quadratic in particle count. Do not compute 300 unreported
    # time steps when only five horizons are needed; retain exact P-pair sum.
    horizons=[t for t in horizons if t<=truth.shape[1]]
    ix=np.asarray(horizons)-1
    score,_=metrics(pred[:,:,ix],truth[:,ix],failed[:,:,ix],horizons=range(1,len(ix)+1))
    return {str(t):dict(score[str(j+1)],seconds=DT*t) for j,t in enumerate(horizons)}


def chunked_checks(checker,history,pred,boundary,states,failed,chunk=4):
    audits=[verify_checks(checker,history[i:i+chunk],pred[i:i+chunk],boundary[i:i+chunk],
                          states[i:i+chunk],failed[i:i+chunk]) for i in range(0,len(pred),chunk)]
    excess=[a['maximum_excess'] for a in audits if a['maximum_excess'] is not None]
    return dict(checked_points=sum(a['checked_points'] for a in audits),
                violations=sum(a['violations'] for a in audits),maximum_excess=max(excess) if excess else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--particles',type=int,default=32)
    p.add_argument('--model-seed',type=int,default=0);args=p.parse_args()
    if args.particles<1:raise ValueError('positive particles required')
    root=Path('adaptive_search_results');w=tail_windows('dev',300,8);runs=[]
    manifest=None if args.model_seed==0 else str(root/f'windows_uniform_seed{args.model_seed}_search_adapter.json')
    stem=f'checked_p{args.particles}_model{args.model_seed}'
    for seed in [1729,2718,3141]:
        cpu=time.process_time();model=CheckedMachine()
        if manifest:model.machine=StatelessPolicyAdapter(model.machine,manifest)
        pred,info=model.rollout(w['history'],300,particles=args.particles,seed=seed)
        failed,states=info['failed'],info['next_state']
        boundary=np.asarray(model.last_stats['last_unreturned_point']).reshape(len(pred),args.particles,2)
        rollout_cpu=time.process_time()-cpu
        verification=chunked_checks(model.checker,w['history'],pred,boundary,states,failed)
        stats={k:v for k,v in model.last_stats.items() if not isinstance(v,list)}
        stats.update(cpu_seconds=time.process_time()-cpu,rollout_cpu_seconds=rollout_cpu)
        scoring_start=time.process_time();score=horizon_metrics(pred,w['truth'],failed)
        per_video={str(v):horizon_metrics(pred[w['video']==v],w['truth'][w['video']==v],failed[w['video']==v]) for v in np.unique(w['video'])}
        if args.particles==8:
            ref=np.load(root/f'adapted_checked_model{args.model_seed}_roll{seed}.npz')
            for k,a in [('prediction',pred),('failed',failed),('states',states),('boundary',boundary),('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:
                np.testing.assert_array_equal(a,ref[k])
        runs.append(dict(seed=seed,score=score,per_video=per_video,verification=verification,stats=stats,
                         failed_endpoints=int(failed[:,:,-1].sum()),total_particles=int(np.prod(failed.shape[:2])),
                         scoring_cpu_seconds=time.process_time()-scoring_start))
        np.savez_compressed(root/f'{stem}_roll{seed}.npz',prediction=pred,failed=failed,states=states,boundary=boundary,
                            truth=w['truth'],video=w['video'],window_start=w['start'])
        report=dict(config=vars(args),manifest=manifest,runs=runs,p8_baseline_bitwise_equal=args.particles==8,
            note='Same32DEV windows, original checked budget_factor10 and sampling; no revision-window2 restriction as in search. Particle streams are not nested across P. No fit/TEST. CPU includes independent checks but excludes scoring/compression, matching prior timing scope. Horizon-only energy uses all P squared pairs, not a sampled approximation.')
        (root/f'{stem}.json').write_text(json.dumps(report,indent=2))
        print(args.particles,args.model_seed,seed,round(stats['cpu_seconds'],3),
              [round(score[str(t)]['embedding_rmse'],6) for t in [50,100,300]],runs[-1]['failed_endpoints'],flush=True)


if __name__=='__main__':main()
