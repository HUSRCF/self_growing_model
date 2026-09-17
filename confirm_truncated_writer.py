"""Eight fresh RNG checks of all frozen full/truncated candidates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from ensemble_score_objective import energy_costs
from validate_truncated_writer import load_models
from v20_rnn_mixture.engine.common import SPLITS


def window_costs(pred,truth,failed):
    costs=[]
    for t in [50,100,300]:
        x=pred[:,:,t-1];y=truth[:,t-1]
        c,_=energy_costs(np.concatenate([np.sin(x),np.cos(x)],-1),np.c_[np.sin(y),np.cos(y)],failed[:,:,t-1])
        costs.append(c)
    return np.stack(costs,1).mean(1)


def paired(values,reference):
    values=np.asarray(values);delta=values-np.asarray(reference)
    if len(delta)<2:raise ValueError('Need at least two paired seeds')
    return dict(objective=float(values.mean()),delta=float(delta.mean()),conditional_seed_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                better=int((delta<0).sum()),paired_delta=delta.tolist())


def main():
    root=Path('adaptive_search_results');models,selections=load_models(root)
    files=[root/'hybrid_writer_pilot.json',root/'hybrid_writer_truncated50_pilot.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    old=json.loads((root/'truncated_writer_validation.json').read_text());assert selections==old['selections']
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    # One previous seed, all candidates, exact scorer replay before fresh seeds.
    cache={}
    for name,model in models.items():
        key=(0.,0.) if model is None else tuple(model['value'])
        if key not in cache:cache[key]=rollout(s,w,model,331017)[0]
        assert cache[key]==old['runs'][name][0],name
    runs={name:[] for name in models}
    for seed in range(341017,341025):
        cache={}
        for name,model in models.items():
            key=(0.,0.) if model is None else tuple(model['value'])
            if key not in cache:
                row,p,f=rollout(s,w,model,seed);cost=window_costs(p,w['truth'],f)
                np.testing.assert_allclose(cost.mean(),row['objective'],rtol=0,atol=1e-14)
                row['window_u_cost']=cost.tolist()
                row['per_video_u_cost']={str(v):float(cost[w['video']==v].mean()) for v in np.unique(w['video'])}
                row['artifact']=f'truncated_confirmation_{name}_{seed}.npz'
                np.savez_compressed(root/row['artifact'],prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
                cache[key]=row
            runs[name].append(cache[key]);print(seed,name,cache[key]['objective'],flush=True)
    base=[r['objective'] for r in runs['zero']];summary={}
    for name,rows in runs.items():
        summary[name]=paired([r['objective'] for r in rows],base)
        summary[name]['per_video']={v:paired([r['per_video_u_cost'][v] for r in rows],[r['per_video_u_cost'][v] for r in runs['zero']]) for v in rows[0]['per_video_u_cost']}
    matched={str(seed):paired([r['objective'] for r in runs[f'truncated_{seed}']],[r['objective'] for r in runs[f'full_{seed}']]) for seed in [1901,2718,3141]}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(selections=selections,runs=runs,summary=summary,truncated_vs_matched_full=matched,previous_seed_exact=True,source_hashes=hashes,
                note='Eight new action seeds341017–24 on SAME TRAINhold24/P8/300,all frozen candidates,not independent videos. Per-video scores are true U-energy composites,not legacy V-energy. No reselection/refit/hyperparameter changes/DEV/TEST; conditional RNG SE only. Artifacts with same parameters share a cache entry.')
    (root/'truncated_writer_confirmation.json').write_text(json.dumps(report,indent=2))
    print({k:{x:v[x] for x in ['objective','delta','conditional_seed_se','better']} for k,v in summary.items()},flush=True)


if __name__=='__main__':main()
