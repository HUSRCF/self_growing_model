"""TRAIN-only periodic kernel fit; frozen output-only smoothing on TRAIN holdout."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import i0e, i1e
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from ensemble_score_objective import energy_costs
from periodic_output_kernel import fit_kernel, nll_gradient, convolve
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def score_endpoints(points, truth, failed):
    return np.stack([energy_costs(embedding(points[:, :, t]), embedding(truth[:, t]), failed[:, :, t])[0]
                     for t in range(3)], 1)


def main():
    root = Path('adaptive_search_results'); files = [Path('v20_rnn_mixture/models/frozen_dynamics.json'), Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    engine = AdaptiveBeam(); fit = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    p, f = continuation(engine, fit['history'], 482017, particles=32)
    if f.any():
        raise RuntimeError('Fit trajectories contain guards; predeclared pilot stops rather than excluding samples')
    centers = p[:, :, [49, 99, 299]]; truth = fit['truth'][:, [49, 99, 299]]
    fits = [fit_kernel(centers[:, :, t], truth[:, t]) for t in range(3)]
    kappa = np.array([a['kappa'] for a in fits])
    print('frozen kappa', kappa.tolist(), 'fit NLL', [a['nll'] for a in fits], flush=True)
    # Kernels fixed before hold data are loaded. Concentration is not refitted.
    hold = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    assert not set(fit['video']) & set(hold['video'])
    from continuous_residual_pilot import rollout
    tiny = {key: value[:1] for key, value in hold.items()}
    _, reference, rf = rollout(engine, tiny, None, 483017, particles=32)
    check, cf = continuation(engine, tiny['history'], 483017, particles=32)
    np.testing.assert_array_equal(check, reference); np.testing.assert_array_equal(cf, rf)
    truth = hold['truth'][:, [49, 99, 299]]; runs = []
    for seed in range(483017, 483021):
        prediction, failed = continuation(engine, hold['history'], seed, particles=32)
        points = prediction[:, :, [49, 99, 299]]; dead = failed[:, :, [49, 99, 299]]
        original = points.copy(); baseline = score_endpoints(points, truth, dead)
        np.testing.assert_array_equal(convolve(points, None, seed+1000000), points)
        changed = convolve(points, kappa, seed+1000000)
        np.testing.assert_array_equal(points, original)
        cost = score_endpoints(changed, truth, dead); delta = cost-baseline
        row = dict(seed=seed, baseline=float(baseline.mean()), kernel=float(cost.mean()), delta=float(delta.mean()),
                   horizon_delta=delta.mean(0).tolist(), baseline_horizons=baseline.mean(0).tolist(), kernel_horizons=cost.mean(0).tolist(),
                   window_delta=delta.mean(1).tolist(), per_video_delta={str(v): float(delta[hold['video']==v].mean()) for v in np.unique(hold['video'])},
                   kernel_nll=[nll_gradient(np.log(kappa[t]), points[:, :, t], truth[:, t])[0] for t in range(3)],
                   guard_particles=int(failed.any(-1).sum()))
        runs.append(row); print('hold', seed, row['delta'], row['horizon_delta'], flush=True)
    delta = np.array([r['delta'] for r in runs])
    summary = dict(baseline=float(np.mean([r['baseline'] for r in runs])), kernel=float(np.mean([r['kernel'] for r in runs])),
                   delta=float(delta.mean()), conditional_seed_se=float(delta.std(ddof=1)/2), better_seeds=int((delta < 0).sum()),
                   horizon_delta=np.mean([r['horizon_delta'] for r in runs], 0).tolist(),
                   per_video_delta={v: float(np.mean([r['per_video_delta'][v] for r in runs])) for v in runs[0]['per_video_delta']},
                   hold_kernel_nll=np.mean([r['kernel_nll'] for r in runs], 0).tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p, h in hashes.items())
    report = dict(fits=fits, kappa=kappa.tolist(), circular_mean_resultant=(i1e(kappa)/i0e(kappa)).tolist(),
                  uniform_nll=float(2*np.log(2*np.pi)), runs=runs, summary=summary, source_hashes=hashes,
                  fit_video=fit['video'].tolist(), fit_start=fit['start'].tolist(), video=hold['video'].tolist(), start=hold['start'].tolist(),
                  original_rollout_exact=True, zero_kernel_exact=True,
                  note='Output ONLY independent product-von-Mises kernels centered at original forecast angles. 3horizons x2 concentrations,TRAINfit80windows seed481017/original32particles seed482017,finite-mixture NLL,one logk start10/bounds1e-4..1e6/200iterations,no hyperparameter search. Frozen before hold24prefix/P32/new483017-20 with independent noise seed+1000000. Original dynamics/event/history/particle weights untouched;zero kernel exact. Proper U-energy scored with ONE noise draw per independent base particle,not correlated flattened replicates. NLL compares finite continuous density only;point baseline has no continuous density,NLL cannot establish improvement over points. Independent horizon noise specifies marginals,not physically consistent joint dynamics. No DEV/TEST/default promotion.')
    (root/'periodic_output_kernel.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
