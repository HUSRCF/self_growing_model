"""Cross-video residual calibration; one velocity draw per trajectory, TRAIN only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_pilot import (prefix_sequence, initializer_data, fit_initializer,
                                   load_block, rollout_memory)
from velocity_memory_compat import LegacyFeatureBridge, legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def residual_covariance(residuals):
    # Each video contributes equal mass, independent of its frame count.
    mean = np.mean([r.mean(0) for r in residuals], axis=0)
    second = np.mean([r.T @ r / len(r) for r in residuals], axis=0)
    cov = second - np.outer(mean, mean)
    return mean, (cov + cov.T) / 2


class InitialVelocityDistribution:
    def __init__(self, writer, covariance, seed):
        self.writer = writer
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ValueError('finite 2x2 covariance required')
        if not np.allclose(covariance, covariance.T):
            raise ValueError('symmetric covariance required')
        values, vectors = np.linalg.eigh(covariance)
        if values.min() < -1e-12:
            raise ValueError('positive semidefinite covariance required')
        self.factor = vectors @ np.diag(np.sqrt(np.maximum(values, 0)))
        self.seed = seed

    def initialize(self, h):
        value = self.writer.initialize(h)
        if not self.factor.any():
            return value
        # Separate RNG: never consumes the event/transition random stream.
        noise = np.random.default_rng(self.seed).standard_normal(value.shape)
        return value + noise @ self.factor.T

    def execute(self, h, q, r, v):
        return self.writer.execute(h, q, r, v)


def main():
    root = Path('adaptive_search_results')
    path = root / 'prefix_velocity_memory_model.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    s = AdaptiveBeam(); bridge = LegacyFeatureBridge(s.base)
    _, Writer = legacy_types()
    sequences = [(v, prefix_sequence(load_video(v))) for v in SPLITS['train'][:-3]]
    residuals = []; folds = []
    for video, y in sequences:
        fitting = [(v, z) for v, z in sequences if v != video]
        init, _ = fit_initializer(bridge, fitting)
        raw, target, last = initializer_data(bridge, [(video, y)])
        residual = target - (last + raw / init['scale'] @ init['coef'])
        residuals.append(residual)
        folds.append(dict(video=video, fit_videos=[v for v, _ in fitting], n=len(raw),
                          proxy_rmse=float(np.sqrt(np.mean(residual**2))),
                          mean=residual.mean(0).tolist()))
    mean, cov = residual_covariance(residuals)
    weak = Writer(bridge, load_block(path), dict(learned_initialization=True))
    hold = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    runs = {name: [] for name in ['original', 'point_velocity', 'distributed_velocity']}
    for seed in range(151017, 151021):
        writers = dict(original=None, point_velocity=weak,
                       distributed_velocity=InitialVelocityDistribution(weak, cov, seed + 1000000))
        for name, writer in writers.items():
            result, p, f, _ = rollout_memory(s, hold, writer, seed)
            runs[name].append(result)
            np.savez_compressed(root / f'initial_velocity_distribution_{name}_{seed}.npz',
                                prediction=p, failed=f, truth=hold['truth'],
                                video=hold['video'], window_start=hold['start'])
            print(name, seed, result['objective'], flush=True)
        if seed == 151017:
            _, p0, f0, _ = rollout_memory(s, hold, InitialVelocityDistribution(weak, np.zeros((2,2)), 42), seed)
            with np.load(root / f'initial_velocity_distribution_point_velocity_{seed}.npz') as old:
                np.testing.assert_array_equal(p0, old['prediction'])
                np.testing.assert_array_equal(f0, old['failed'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    comparisons = {}
    for reference in ['original', 'point_velocity']:
        d = np.array([a['objective']-b['objective'] for a,b in zip(runs['distributed_velocity'], runs[reference])])
        comparisons[reference] = dict(differences=d.tolist(), mean=float(d.mean()),
            conditional_rng_se=float(d.std(ddof=1)/np.sqrt(len(d))))
    report = dict(folds=folds, calibration_mean_not_applied=mean.tolist(), covariance=cov.tolist(),
        runs=runs, comparisons=comparisons, model_sha256=digest, model_unchanged=True,
        zero_covariance_exact=True, mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        note='TRAIN prefixes only; leave-one-fit-video-out initializer residual calibration; equal video weights, centered covariance, no mean shift, fixed scale1. One independent Gaussian initial velocity draw per particle; no step noise. Smoothed proxy residual covariance is not a physical posterior. Fixed models/new action seeds on reused TRAIN hold videos, not independent video validation. No DEV/TEST/promotion.')
    (root / 'initial_velocity_distribution.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
