"""Fixed protected writer DEV evaluation; no fitting or parameter selection."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from ensemble_score_objective import energy_costs
from v20_rnn_mixture.engine.data import tail_windows


def distribution_scores(pred,truth,failed,video):
    result={}
    for t in [50,100,300]:
        x=pred[:,:,t-1];y=truth[:,t-1]
        emb=np.concatenate([np.sin(x),np.cos(x)],-1);target=np.c_[np.sin(y),np.cos(y)]
        cost,_=energy_costs(emb,target,failed[:,:,t-1])
        result[str(t)]=dict(mean=float(cost.mean()),
            per_video={str(v):float(cost[video==v].mean()) for v in np.unique(video)})
    return result


def main():
    root=Path('adaptive_search_results');path=root/'protected_residual_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    s=AdaptiveBeam();w=tail_windows('dev',300,8);runs={'zero':[],'protected':[]}
    for seed in [1729,2718,3141]:
        for name,m in [('zero',None),('protected',model)]:
            start=time.process_time();r,p,f=rollout(s,w,m,seed)
            r['cpu_seconds']=time.process_time()-start
            r['energy_u']=distribution_scores(p,w['truth'],f,w['video'])
            if name=='zero':
                with np.load(root/f'feedback_distribution_pilot_frozen_point_{seed}.npz') as previous:
                    np.testing.assert_array_equal(p,previous['prediction'])
                    np.testing.assert_array_equal(f,previous['failed'])
                    for key,value in [('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:
                        np.testing.assert_array_equal(value,previous[key])
            np.savez_compressed(root/f'protected_residual_dev_{name}_{seed}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
            runs[name].append(r);print(name,seed,r['objective'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    summary={name:dict(objective=float(np.mean([r['objective'] for r in rows])),
             horizons={str(t):{key:float(np.mean([r['score'][str(t)][key] for r in rows]))
                       for key in ['embedding_rmse','energy_score','coverage90','failure']} for t in [50,100,300]},
             energy_u={str(t):float(np.mean([r['energy_u'][str(t)]['mean'] for r in rows])) for t in [50,100,300]})
             for name,rows in runs.items()}
    report=dict(runs=runs,summary=summary,model_sha256=digest,model_unchanged=True,
                baseline_exact_replay=True,
                note='Fixed TRAIN-fit scale/model,32reused DEV tail windows/P8/3actionseeds; no tuning or TEST. No checker/search enforcement; numeric completion not validity. V energy is finite-ensemble metric, U excludes diagonal with conditional IID caveats. Seeds/windows not independent videos. CPU rollout includes existing diagnostics/scoring, excludes added U per-video scoring and compression.')
    (root/'protected_residual_dev.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
