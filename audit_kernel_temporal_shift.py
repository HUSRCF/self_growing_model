"""Frozen kernels on same hold videos' later windows, with paired prefix reports."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from validate_energy_output_kernel import evaluate


def main():
    root = Path('adaptive_search_results')
    source = root/'energy_output_kernel_confirmation.json'
    old = json.loads(source.read_text())
    hashes = dict(old['source_hashes'])
    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    kernels = old['kernels']; runs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(evaluate, row['seed'], kernels, 64, 'tail') for row in old['runs']]
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('tail complete', row['seed'], row['gates'], flush=True)
    runs.sort(key=lambda row: row['seed'])
    video = np.array(runs[0]['video'])
    for row, prefix in zip(runs, old['runs']):
        assert row['seed'] == prefix['seed']
        np.testing.assert_array_equal(row['video'], prefix['video'])
        np.testing.assert_array_equal(row['start'], runs[0]['start'])
        # Entire tail truth begins after every prefix target in the same video.
        for v in np.unique(video):
            mask = video == v
            assert np.array(row['start'])[mask].min()+1 > (np.array(prefix['start'])[mask]+300).max()
    zero = np.array([r['results']['4097']['zero'] for r in runs])
    summary = dict(zero=float(zero.mean()))
    for name in kernels:
        value = np.array([r['results']['4097'][name]['cost'] for r in runs])
        delta = value-zero; means = delta.mean((1, 2))
        summary[name] = dict(score=float(value.mean()), delta=float(delta.mean()), seed_deltas=means.tolist(),
                             conditional_seed_se=float(means.std(ddof=1)/np.sqrt(len(means))),
                             better_seeds=int((means < 0).sum()), horizon_delta=delta.mean((0, 1)).tolist(),
                             per_video_delta={str(v): float(delta[:, video == v].mean()) for v in np.unique(video)},
                             tail_minus_prefix_delta=float(means.mean()-old['summary'][name]['delta']))
    summary['energy_minus_nll32'] = float(summary['energy']['score']-summary['nll32']['score'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(kernels=kernels, source_hashes=hashes, runs=runs, summary=summary, prefix_summary=old['summary'],
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for r in runs for g in r['gates'].values()),
                  note='Frozen U/NLL32/zero,TRAINhold18/16/13 tail8each,P64,same521017-24 as saved prefix. '
                       'Targets disjoint between regions,within-region windows can overlap and firsttail history crosses boundary. '
                       'Same videos/RNG not new independent videos;time-region association not causal decomposition. '
                       'Allwindows2049/4097 score1e-4/gradient1e-5;8workersBLAS1. '
                       'No refit/selection/DEV/TEST/promotion;hold/backboneTRAIN historically used.')
    (root/'energy_kernel_temporal_shift.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, all_precision_gates_pass=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
