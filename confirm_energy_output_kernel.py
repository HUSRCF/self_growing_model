"""Higher-particle frozen confirmation; no parameter updates or model selection."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from validate_energy_output_kernel import evaluate
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from spectral_kernel_energy import blocked_value


def main():
    root = Path('adaptive_search_results')
    source = root/'energy_output_kernel_holdout.json'
    old = json.loads(source.read_text())
    hashes = dict(old['source_hashes'])
    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    kernels = old['kernels']
    hold = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    reference = old['runs'][0]
    p, f = continuation(AdaptiveBeam(), hold['history'], reference['seed'], particles=32)
    baseline = score_endpoints(p[:, :, [49, 99, 299]], hold['truth'][:, [49, 99, 299]], f[:, :, [49, 99, 299]])
    np.testing.assert_array_equal(baseline, reference['results']['4097']['zero'])
    for size in [2049, 4097]:
        for name, kappa in kernels.items():
            for t, step in enumerate([49, 99, 299]):
                row = blocked_value(p[0, :, step], hold['truth'][0, step], f[0, :, step], size, np.log(kappa[t]))
                assert row['cost'] == reference['results'][str(size)][name]['cost'][0][t]
                np.testing.assert_array_equal(row['gradient'], reference['results'][str(size)][name]['gradient'][0][t])
    print('old baseline all24 and first-window kernels exact', flush=True)
    runs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(evaluate, seed, kernels, 64) for seed in range(521017, 521025)]
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('seed complete', row['seed'], row['gates'], flush=True)
    runs.sort(key=lambda row: row['seed'])
    video = np.array(reference['video'])
    for row in runs:
        np.testing.assert_array_equal(row['video'], reference['video'])
        np.testing.assert_array_equal(row['start'], reference['start'])
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
                  old_baseline_exact=True, old_first_window_kernel_exact=True,
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for r in runs for g in r['gates'].values()),
                  note='Frozen3models,sameTRAINhold24prefix,new521017-24,P64vsoldP32. '
                       'EightworkersBLAS1.4x trajectory sampling budget,not equal compute comparison. '
                       'Allwindows2049/4097score1e-4/grad1e-5 gates,no addedkernelnoiseMC. '
                       'No refit/selection/DEV/TEST/promotion. Historically reusedhold/backboneTRAIN; '
                       'seedSE conditional on fixedwindows,not videoCI.')
    (root/'energy_output_kernel_confirmation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, all_precision_gates_pass=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
