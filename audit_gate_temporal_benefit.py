"""Frozen five-strategy error audit on fitting videos, not temporal cross-validation."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_gate_temporal_coverage import region_starts
from training_window_sampler import TrainingPrefixPool
from validate_soft_gates_new_windows import causal_features
from validate_full_soft_gates import weights
from crossfit_soft_kernel_gate import coefficients
from soft_output_mixture import value_gradient
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def prepare(region, seed, model, excluded):
    engine = AdaptiveBeam()
    histories, truths, videos, starts = [], [], [], []
    for video in SPLITS['train'][:-3]:
        y = load_video(video)
        chosen = region_starts(len(y), region, excluded[str(video)], count=8)
        for t in chosen:
            histories.append(y[t-31:t+1]); truths.append(y[t+1:t+301])
            videos.append(video); starts.append(int(t))
    w = dict(history=np.asarray(histories))
    motion, raw = causal_features(engine, w)
    alpha = weights(model, motion, raw)
    p, f = continuation(engine, w['history'], seed, particles=32)
    return dict(region=region, seed=seed, video=videos, start=starts, motion=motion,
                points=p[:, :, [49, 99, 299]], truth=np.asarray(truths)[:, [49, 99, 299]],
                failed=f[:, :, [49, 99, 299]], guards=int(f.any(-1).sum()), alpha=alpha)


def score_chunk(kernel, row, begin):
    s = slice(begin, begin+20)
    terms, precision = coefficients(kernel, row['motion'][s], row['points'][s], row['truth'][s], row['failed'][s])
    fine = {k: np.asarray(v) for k, v in terms['4097'].items()}
    return dict(region=row['region'], seed=row['seed'], begin=begin,
                coefficients=terms, precision=precision,
                costs={name: value_gradient(fine, a[s])[0].tolist() for name, a in row['alpha'].items()})


def main():
    root = Path('adaptive_search_results')
    paths = [root/'full_soft_gate_model.json', Path('v20_rnn_mixture/models/frozen_dynamics.json'),
             Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model = json.loads(paths[0].read_text())['model']
    fit = TrainingPrefixPool(AdaptiveBeam().base, steps=300).sample(481017, per_video=8)
    excluded = {str(v): fit['start'][fit['video'] == v].tolist() for v in SPLITS['train'][:-3]}
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(prepare, region, seed, model, excluded)
                 for region in ['prefix', 'tail'] for seed in [611017, 611018]]
        prepared = [job.result() for job in tasks]
    print('four full-batch rollouts ready; frozen weights', flush=True)
    chunks = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks = [pool.submit(score_chunk, model['kernel'], row, i) for row in prepared for i in range(0, 80, 20)]
        for job in as_completed(tasks):
            chunk = job.result(); chunks.append(chunk)
            print('scored', len(chunks), 'of', len(tasks), chunk['precision'], flush=True)
    runs = []
    for row in prepared:
        matching = sorted([r for r in chunks if (r['region'], r['seed']) == (row['region'], row['seed'])], key=lambda r: r['begin'])
        runs.append(dict(region=row['region'], seed=row['seed'], video=row['video'], start=row['start'],
                         motion=row['motion'].tolist(), guards=row['guards'],
                         alpha={k: v.tolist() for k, v in row['alpha'].items()},
                         costs={k: np.concatenate([r['costs'][k] for r in matching]).tolist() for k in row['alpha']}))
    summary = {}
    for region in ['prefix', 'tail']:
        selected = [r for r in runs if r['region'] == region]
        video = np.asarray(selected[0]['video'])
        costs = {k: np.asarray([r['costs'][k] for r in selected]) for k in selected[0]['costs']}
        summary[region] = dict(zero=float(costs['zero'].mean()))
        for name in ['full', 'scalar', 'contextual', 'features']:
            delta = costs[name]-costs['zero']
            summary[region][name] = dict(score=float(costs[name].mean()), delta=float(delta.mean()),
                                        seed_deltas=delta.mean((1, 2)).tolist(), horizon_delta=delta.mean((0, 1)).tolist(),
                                        per_video_delta={str(v): float(delta[:, video == v].mean()) for v in np.unique(video)})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(summary=summary, runs=runs, chunks=chunks, source_hashes=hashes,
                  all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in chunks),
                  note='Frozen fullTRAIN gates/kernel, fitting10 only,8 evenly spaced new starts/video/region,611017/18 P32. '
                       'No fitting/selection/hold/DEV/TEST. Prefix may overlap fitted targets; tail histories wholly after midpoint. '
                       'Descriptive frozen-model temporal transfer, NOT cross-validation; backbone TRAIN and within-region overlap. '
                       'All five share trajectories; exact-point uniform-alpha2049/4097 gates unchanged; failures preserved.')
    (root/'gate_temporal_benefit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, precision=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
