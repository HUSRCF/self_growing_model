"""Frozen equal-gradient-sampling-budget first Adam updates, TRAIN only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_writer import constant_model
from confirm_truncated_writer import window_costs, paired


def first_step(gradient):
    g = np.asarray(gradient, dtype=float)
    if g.shape != (8, 2) or not np.isfinite(g).all():
        raise ValueError('Expected finite 8x2 gradient')
    theta = torch.zeros((8, 2), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=.1)
    theta.grad = torch.tensor(g.copy(), dtype=torch.float64)
    torch.nn.utils.clip_grad_norm_([theta], 1.)
    optimizer.step()
    return theta.detach().numpy().clip(-1, 1)


def candidates(report):
    a = np.array([r['gradients'] for r in report['phases']['fit']])
    b = np.array([r['gradients'] for r in report['phases']['evaluation']])
    s = np.array([r['score_gradients'] for r in report['phases']['evaluation']])
    assert a.shape == b.shape == s.shape == (8, 10, 8, 2)
    coefficients = np.asarray(report['coefficients'])[None, :, None, None]
    gradients = {'raw8': a.mean((0, 1)),
                 'raw16': np.concatenate([a, b]).mean((0, 1)),
                 'controlled16': (b - coefficients * s).mean((0, 1))}
    gradients['reverse_raw16'] = -gradients['raw16']
    return {'zero': np.zeros((8, 2)), **{k: first_step(g) for k, g in gradients.items()}}, gradients


def main():
    torch.set_num_threads(1)
    root = Path('adaptive_search_results')
    source = root / 'score_control_variate_audit.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    previous = json.loads(source.read_text())
    parameters, gradients = candidates(previous)
    with np.load(root / 'continuous_residual_model_ridge_0.0001.npz') as f:
        bound = f['cap'].copy() * .01
    engine = AdaptiveBeam()
    pool = TrainingPrefixPool(engine.base, steps=300)
    regions = {'same_windows': pool.sample(190101, per_video=1),
               'fresh_windows': pool.sample(392017, per_video=1)}
    np.testing.assert_array_equal(regions['same_windows']['start'], previous['window_start'])
    np.testing.assert_array_equal(regions['same_windows']['video'], previous['video'])
    runs, summary, metadata = {}, {}, {}
    for region, w in regions.items():
        metadata[region] = {k: w[k].tolist() for k in ['video', 'start']}
        rows = {name: [] for name in parameters}
        for seed in range(391017, 391021):
            for name, theta in parameters.items():
                model = constant_model(theta, bound)
                model['kind'] = 'source_q'
                row, prediction, failed = rollout(engine, w, model, seed, particles=4)
                cost = window_costs(prediction, w['truth'], failed)
                np.testing.assert_allclose(cost.mean(), row['objective'], rtol=0, atol=1e-14)
                rows[name].append(dict(seed=seed, objective=row['objective'],
                                       window_costs=cost.tolist(), failed_particles=int(failed.any(-1).sum())))
                print(region, seed, name, row['objective'], flush=True)
        base = [r['objective'] for r in rows['zero']]
        runs[region] = rows
        summary[region] = {name: paired([r['objective'] for r in rs], base) for name, rs in rows.items()}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(parameters={k: v.tolist() for k, v in parameters.items()},
                  gradients={k: v.tolist() for k, v in gradients.items()}, bound=bound.tolist(),
                  metadata=metadata, runs=runs, summary=summary, source_sha256=digest,
                  note='Frozen before rollouts: Adam first step lr .1/eps1e-8/clip1/box1. Raw16 and controlled16 each use16 gradient batches (control8fit+8evaluation); raw8 cheaper. Previous evaluation gradients now reused as TRAIN fitting data. New forward RNG391017-20/P4/300; same and fresh TRAIN-fit windows (190101/392017); windows may overlap in time, not independent videos. Truncated50 estimator, no unbiased full-gradient claim. No tuning/selection/hold/DEV/TEST/default promotion. Conditional RNG SE only; equal sampling budget not total CPU.')
    (root / 'first_score_update.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
