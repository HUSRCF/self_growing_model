"""Frozen U/NLL32/zero output kernels, paired TRAIN-holdout seeds and grid gates."""
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from spectral_kernel_energy import blocked_value
from v20_rnn_mixture.engine.data import tail_windows


def evaluate(seed, kernels, particles=32, region='prefix'):
    start = time.process_time()
    if region == 'prefix':
        hold = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    elif region == 'tail':
        all_tail = tail_windows('train', 300, 8)
        mask = np.isin(all_tail['video'], SPLITS['train'][-3:])
        hold = {key: value[mask] for key, value in all_tail.items()}
    else:
        raise ValueError('Unknown temporal region')
    p, f = continuation(AdaptiveBeam(), hold['history'], seed, particles=particles)
    results = {}
    for size in [2049, 4097]:
        data = {name: dict(cost=np.zeros((24, 3)), gradient=np.zeros((24, 3, 2))) for name in kernels}
        baseline = np.zeros((24, 3))
        for j in range(24):
            for t, step in enumerate([49, 99, 299]):
                for name, kappa in kernels.items():
                    row = blocked_value(p[j, :, step], hold['truth'][j, step], f[j, :, step], size, np.log(kappa[t]))
                    data[name]['cost'][j, t] = row['cost']
                    data[name]['gradient'][j, t] = row['gradient']
                    baseline[j, t] = row['baseline']
        results[str(size)] = {name: {key: value.tolist() for key, value in row.items()} for name, row in data.items()}
        results[str(size)]['zero'] = baseline.tolist()
    np.testing.assert_array_equal(results['2049']['zero'], results['4097']['zero'])
    gates = {}
    for name in kernels:
        errors = {key: float(np.max(np.abs(np.array(results['2049'][name][key])-results['4097'][name][key])))
                  for key in ['cost', 'gradient']}
        gates[name] = dict(**errors, cost_pass=errors['cost'] <= 1e-4, gradient_pass=errors['gradient'] <= 1e-5)
    return dict(seed=seed, results=results, gates=gates, cpu_seconds=time.process_time()-start,
                guards=int(f.any(-1).sum()), video=hold['video'].tolist(), start=hold['start'].tolist())


def main():
    root = Path('adaptive_search_results')
    files = [root/'energy_output_kernel_fit.json', root/'periodic_output_kernel.json',
             Path('v20_rnn_mixture/models/frozen_dynamics.json'), Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fit, nll = [json.loads(p.read_text()) for p in files[:2]]
    kernels = dict(energy=[x['kappa'] for x in fit['fits']], nll32=nll['kappa'])
    assert all(x['success'] for x in fit['fits'])
    # All kernels are fixed before any worker loads holdout data.
    runs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(evaluate, seed, kernels): seed for seed in range(511017, 511021)}
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('seed complete', row['seed'], row['gates'], flush=True)
    runs.sort(key=lambda r: r['seed'])
    for row in runs:
        np.testing.assert_array_equal(row['video'], runs[0]['video'])
        np.testing.assert_array_equal(row['start'], runs[0]['start'])
    video = np.array(runs[0]['video'])
    zero = np.array([r['results']['4097']['zero'] for r in runs])
    summary = dict(zero=float(zero.mean()))
    for name in kernels:
        value = np.array([r['results']['4097'][name]['cost'] for r in runs])
        delta = value-zero; means = delta.mean((1, 2))
        summary[name] = dict(score=float(value.mean()), delta=float(delta.mean()), seed_deltas=means.tolist(),
                             conditional_seed_se=float(means.std(ddof=1)/np.sqrt(len(means))),
                             better_seeds=int((means < 0).sum()), horizon_delta=delta.mean((0, 1)).tolist(),
                             per_video_delta={str(v): float(delta[:, video == v].mean()) for v in np.unique(video)})
    summary['energy_minus_nll32'] = float(summary['energy']['score']-summary['nll32']['score'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(kernels=kernels, source_hashes=hashes, runs=runs, summary=summary,
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for r in runs for g in r['gates'].values()),
                  note='Frozen U/NLL32/zero, same original32 fit centers but different objectives. '
                       'TRAINhold24prefix18/16/13,P32,new511017-20. Four independent CPU workers,BLAS1. '
                       'Deterministic independent-kernel-noise U integration,all window2049/4097 gates1e-4/1e-5. '
                       'No fitting/selection/DEV/TEST/promotion. Hold videos historically reused and backbone TRAIN; '
                       'not global blind test. Seed SE conditional on fixed windows,not video generalization uncertainty.')
    (root/'energy_output_kernel_holdout.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, all_precision_gates_pass=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
