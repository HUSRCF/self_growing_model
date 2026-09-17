"""Fixed-covariance crossed action/initialization RNG audit, TRAIN hold only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from initial_velocity_distribution import InitialVelocityDistribution
from velocity_memory_pilot import load_block, rollout_memory
from velocity_memory_compat import LegacyFeatureBridge, legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def crossed_summary(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('finite crossed matrix with at least two levels per factor required')
    mean = x.mean(); row = x.mean(1); col = x.mean(0)
    residual = x - row[:,None] - col[None,:] + mean
    ss = dict(action=float(x.shape[1]*np.sum((row-mean)**2)),
              initialization=float(x.shape[0]*np.sum((col-mean)**2)),
              interaction=float(np.sum(residual**2)))
    total = float(np.sum((x-mean)**2))
    return dict(mean=float(mean), action_means=row.tolist(), initialization_means=col.tolist(),
                sum_squares=ss, total_sum_squares=total,
                descriptive_fractions={k:v/total if total else 0. for k,v in ss.items()},
                note='Balanced-table descriptive decomposition, not independent-cell SE or population variance components.')


def main():
    root = Path('adaptive_search_results')
    source = root/'initial_velocity_distribution.json'
    model = root/'prefix_velocity_memory_model.npz'
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,model]}
    cov = np.array(json.loads(source.read_text())['covariance'])
    s = AdaptiveBeam(); _, Writer = legacy_types()
    weak = Writer(LegacyFeatureBridge(s.base), load_block(model), dict(learned_initialization=True))
    w = prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    _,p,f,_ = rollout_memory(s,w,InitialVelocityDistribution(weak,cov,1151017),151017)
    with np.load(root/'initial_velocity_distribution_distributed_velocity_151017.npz') as old:
        np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
    actions=list(range(161017,161021)); initials=list(range(1161017,1161021))
    baseline={'original':[],'point_velocity':[]}; distributed=[]
    for action in actions:
        for name,writer in [('original',None),('point_velocity',weak)]:
            r,p,f,_=rollout_memory(s,w,writer,action);baseline[name].append(r)
            np.savez_compressed(root/f'initial_rng_{name}_{action}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
        row=[]
        for init in initials:
            r,p,f,_=rollout_memory(s,w,InitialVelocityDistribution(weak,cov,init),action)
            r['initialization_seed']=init;row.append(r)
            np.savez_compressed(root/f'initial_rng_distributed_{action}_{init}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
            print(action,init,r['objective'],flush=True)
        distributed.append(row)
    matrix=np.array([[r['objective'] for r in row] for row in distributed])
    comparisons={name:crossed_summary(matrix-np.array([r['objective'] for r in rows])[:,None])
                 for name,rows in baseline.items()}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(action_seeds=actions,initialization_seeds=initials,baseline=baseline,
                distributed=distributed,objective_matrix=matrix.tolist(),summary=crossed_summary(matrix),
                comparisons=comparisons,source_hashes=hashes,sources_unchanged=True,original_replay_exact=True,
                note='Fixed calibrated covariance and weights. TRAIN hold24/P8/300, 4x4 fresh RNG crossing. Baselines once per action seed; 16 cells not independent video replicates. No fitting, scale search, DEV/TEST, or promotion.')
    (root/'initial_velocity_rng_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
