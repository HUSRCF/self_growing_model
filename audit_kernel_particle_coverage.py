"""Nested32->128 TRAIN center samples; no bandwidth or hold-driven tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import i0e, i1e
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from periodic_output_kernel import fit_kernel, nll_gradient, convolve
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root = Path('adaptive_search_results'); source = root/'periodic_output_kernel.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); old = json.loads(source.read_text())
    engine = AdaptiveBeam(); fit = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(fit[key], old['fit_'+key])
    truth = fit['truth'][:, [49, 99, 299]]; batches = []; batch_hashes = []
    for seed in range(482017, 482021):
        prediction, failed = continuation(engine, fit['history'], seed, particles=32)
        if failed.any():
            raise RuntimeError('Fit trajectories contain guards; no exclusions allowed')
        centers = prediction[:, :, [49, 99, 299]]
        batches.append(centers); batch_hashes.append(hashlib.sha256(centers.tobytes()).hexdigest())
        if seed == 482017:
            reproduced = [fit_kernel(centers[:, :, t], truth[:, t]) for t in range(3)]
            assert reproduced == old['fits'], 'Original kernel fit must replay exactly'
        print('fit center batch', seed, 'complete', flush=True)
    centers = np.concatenate(batches, axis=1)
    np.testing.assert_array_equal(centers[:, :32], batches[0])
    fits = [fit_kernel(centers[:, :, t], truth[:, t]) for t in range(3)]
    kernels = {'old32': np.array(old['kappa']), 'new128': np.array([r['kappa'] for r in fits])}
    training_nll = {name: [nll_gradient(np.log(k[t]), centers[:, :, t], truth[:, t])[0] for t in range(3)]
                    for name, k in kernels.items()}
    print('frozen kernels', {k: v.tolist() for k, v in kernels.items()}, flush=True)
    hold = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(hold[key], old[key])
    truth = hold['truth'][:, [49, 99, 299]]
    p, f = continuation(engine, hold['history'], 483017, particles=32)
    points, dead = p[:, :, [49, 99, 299]], f[:, :, [49, 99, 299]]
    baseline = score_endpoints(points, truth, dead)
    cost = score_endpoints(convolve(points, kernels['old32'], 1483017), truth, dead)
    assert float(baseline.mean()) == old['runs'][0]['baseline']
    assert float(cost.mean()) == old['runs'][0]['kernel']
    np.testing.assert_array_equal((cost-baseline).mean(1), old['runs'][0]['window_delta'])
    runs = {name: [] for name in ['zero', *kernels]}
    for seed in range(491017, 491021):
        p, f = continuation(engine, hold['history'], seed, particles=32)
        points, dead = p[:, :, [49, 99, 299]], f[:, :, [49, 99, 299]]
        baseline = score_endpoints(points, truth, dead)
        for name in runs:
            kappa = None if name == 'zero' else kernels[name]
            cost = score_endpoints(convolve(points, kappa, seed+1000000), truth, dead)
            if name == 'zero': np.testing.assert_array_equal(cost, baseline)
            delta = cost-baseline
            runs[name].append(dict(seed=seed, cost=float(cost.mean()), delta=float(delta.mean()),
                                   horizon_delta=delta.mean(0).tolist(), window_delta=delta.mean(1).tolist(),
                                   per_video_delta={str(v): float(delta[hold['video']==v].mean()) for v in np.unique(hold['video'])},
                                   nll=None if kappa is None else [nll_gradient(np.log(kappa[t]), points[:, :, t], truth[:, t])[0] for t in range(3)],
                                   guards=int(f.any(-1).sum())))
        print('fresh', seed, {name: rs[-1]['delta'] for name, rs in runs.items()}, flush=True)
    summary = {}
    for name, rows in runs.items():
        delta = np.array([r['delta'] for r in rows])
        summary[name] = dict(cost=float(np.mean([r['cost'] for r in rows])), delta=float(delta.mean()),
                             conditional_seed_se=float(delta.std(ddof=1)/2), better_seeds=int((delta < 0).sum()),
                             horizon_delta=np.mean([r['horizon_delta'] for r in rows], 0).tolist(),
                             per_video_delta={v: float(np.mean([r['per_video_delta'][v] for r in rows])) for v in rows[0]['per_video_delta']})
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(kernels={k: v.tolist() for k, v in kernels.items()}, fits=fits, training_nll_on128=training_nll,
                  mean_resultants={k: (i1e(v)/i0e(v)).tolist() for k, v in kernels.items()}, runs=runs, summary=summary,
                  source_sha256=digest, batch_hashes=batch_hashes, old_fit_exact=True, old_hold_seed_exact=True,
                  fit_video=fit['video'].tolist(), fit_start=fit['start'].tolist(), video=hold['video'].tolist(), start=hold['start'].tolist(),
                  note='Same80TRAINfitwindows481017,old32centers482017 plus3independent32batches482018-20 nested128. Same6kappas/optimizer/start/bounds,no refit on hold.4xfit sampling cost,not equal budget. Oldfit and483017hold score exact replay; old/new/zero frozen before new491017-20 hold24/P32. Noise seed+1e6 same initialization across kernels,but vonMises rejection sampling may consume different streams,not identical perturbations. NLL evaluation remains finite32center approximation,not directly comparable to point mass NLL. Output marginal smoothing only,no event/dynamics feedback/DEV/TEST/promotion.')
    (root/'kernel_particle_coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
