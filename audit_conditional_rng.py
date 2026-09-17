"""Crossed RNG replication for fixed conditional covariance, reusing global runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from conditional_velocity_distribution import ConditionalVelocityDistribution
from audit_initial_velocity_rng import crossed_summary
from audit_conditional_trajectories import energy_parts
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');config=root/'conditional_velocity_distribution.json'
    reference=root/'initial_velocity_rng_audit.json';model=root/'prefix_velocity_memory_model.npz'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [config,reference,model]}
    cfg=json.loads(config.read_text());ref=json.loads(reference.read_text())
    assert hashes[str(model)]==cfg['source_hashes'][str(model)]==ref['source_hashes'][str(model)]
    s=AdaptiveBeam();_,Writer=legacy_types()
    weak=Writer(LegacyFeatureBridge(s.base),load_block(model),dict(learned_initialization=True))
    make=lambda seed:ConditionalVelocityDistribution(weak,cfg['thresholds'],cfg['covariances'],seed)
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    _,p,f,_=rollout_memory(s,w,make(1171017),171017)
    with np.load(root/'conditional_velocity_conditional_171017.npz') as old:
        np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
    runs=[];deltas=[]
    for i,action in enumerate(ref['action_seeds']):
        row=[];delta_row=[]
        for j,init in enumerate(ref['initialization_seeds']):
            path=root/f'initial_rng_distributed_{action}_{init}.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                for key,v in [('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:
                    np.testing.assert_array_equal(z[key],v)
                global_parts=energy_parts(z['prediction'],z['truth'],z['failed'])
            np.testing.assert_allclose(global_parts['total'][:,[49,99,299]].mean(),ref['distributed'][i][j]['objective'],rtol=0,atol=1e-14)
            r,p,f,_=rollout_memory(s,w,make(init),action);r['initialization_seed']=init;row.append(r)
            parts=energy_parts(p,w['truth'],f)
            np.testing.assert_allclose(parts['total'][:,[49,99,299]].mean(),r['objective'],rtol=0,atol=1e-14)
            delta_row.append({k:parts[k]-global_parts[k] for k in parts})
            np.savez_compressed(root/f'conditional_rng_{action}_{init}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
            print(action,init,r['objective']-ref['distributed'][i][j]['objective'],flush=True)
        runs.append(row);deltas.append(delta_row)
    delta={k:np.array([[r[k] for r in row] for row in deltas]) for k in deltas[0][0]}
    # Shape: action, initialization, window, time; do not flatten RNG cells for SE.
    objective=np.array([[r['objective'] for r in row] for row in runs])
    comparisons={'global':crossed_summary(delta['total'][:,:,:,[49,99,299]].mean((2,3)))}
    for name,rows in ref['baseline'].items():
        comparisons[name]=crossed_summary(objective-np.array([r['objective'] for r in rows])[:,None])
    horizons={str(t):{k:crossed_summary(v[:,:,:,t-1].mean(2)) for k,v in delta.items()} for t in [50,100,300]}
    videos={str(v):crossed_summary(delta['total'][:,:,w['video']==v][:,:,:,[49,99,299]].mean((2,3))) for v in np.unique(w['video'])}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(runs=runs,comparisons=comparisons,horizons=horizons,per_video=videos,
        action_seeds=ref['action_seeds'],initialization_seeds=ref['initialization_seeds'],
        conditional_mean_objective=float(objective.mean()),global_mean_objective=ref['summary']['mean'],
        mean_curves={k:v.mean((0,1,2)).tolist() for k,v in delta.items()},
        source_hashes=hashes,sources_unchanged=True,conditional_replay_exact=True,global_scores_replayed=True,
        note='Fixed covariance/means/dynamics, same old4x4 RNG crossing as global reference. No fitting/selection/DEV/TEST. Shared videos/windows and factor levels: 16 cells not independent replications. Late reversal is a hypothesis, not assumed true.')
    (root/'conditional_rng_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
