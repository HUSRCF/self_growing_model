"""Frozen hybrid writer DEV gate; report all candidates, never tune on DEV."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from confirm_truncated_writer import window_costs,paired
from validate_truncated_writer import load_models
from v20_rnn_mixture.engine.data import tail_windows


def main():
    root=Path('adaptive_search_results');models,selections=load_models(root)
    files=[root/'hybrid_writer_pilot.json',root/'hybrid_writer_truncated50_pilot.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    confirmation=json.loads((root/'truncated_writer_confirmation.json').read_text())
    assert selections==confirmation['selections']
    s=AdaptiveBeam();w=tail_windows('dev',300,8);assert len(w['history'])==32
    runs={name:[] for name in models}
    for seed in [1729,2718,3141]:
        cache={}
        for name,model in models.items():
            key=(0.,0.) if model is None else tuple(model['value'])
            if key not in cache:
                row,p,f=rollout(s,w,model,seed)
                if name=='zero':
                    with np.load(root/f'feedback_distribution_pilot_frozen_point_{seed}.npz') as old:
                        for k,v in [('prediction',p),('failed',f),('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:np.testing.assert_array_equal(v,old[k])
                cost=window_costs(p,w['truth'],f)
                np.testing.assert_allclose(cost.mean(),row['objective'],rtol=0,atol=1e-14)
                row['window_u_cost']=cost.tolist();row['failed_endpoint_count']=int(f[:,:,-1].sum())
                row['per_video_u_cost']={str(v):float(cost[w['video']==v].mean()) for v in np.unique(w['video'])}
                row['artifact']=f'hybrid_writer_dev_{name}_{seed}.npz'
                np.savez_compressed(root/row['artifact'],prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
                cache[key]=row
            runs[name].append(cache[key]);print(seed,name,cache[key]['objective'],flush=True)
    base=[r['objective'] for r in runs['zero']];summary={}
    for name,rows in runs.items():
        summary[name]=paired([r['objective'] for r in rows],base)
        summary[name]['per_video']={v:paired([r['per_video_u_cost'][v] for r in rows],[r['per_video_u_cost'][v] for r in runs['zero']]) for v in rows[0]['per_video_u_cost']}
        summary[name]['embedding_rmse']={str(t):float(np.mean([r['score'][str(t)]['embedding_rmse'] for r in rows])) for t in [50,100,300]}
        summary[name]['failed_endpoints']=sum(r['failed_endpoint_count'] for r in rows)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(selections=selections,runs=runs,summary=summary,baseline_exact_replay=True,source_hashes=hashes,
                note='Frozen selections,DEV32 tail windows/P8/300,3fixed seeds1729/2718/3141. All candidates retained; no tuning/reselection/training/TEST/default promotion. DEV historically reused,not blind confirmation. Per-video U-energy differs from legacy V-energy metric. Conditional seed SE not video-generalization uncertainty.')
    (root/'hybrid_writer_dev.json').write_text(json.dumps(report,indent=2));print({k:{s:v[s] for s in ['objective','delta','embedding_rmse','failed_endpoints']} for k,v in summary.items()},flush=True)


if __name__=='__main__':main()
