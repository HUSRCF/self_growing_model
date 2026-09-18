"""Frozen adapter crossfit confirmation, more independent trajectory samples."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from crossfit_output_kernels import evaluate_fold
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from training_window_sampler import TrainingPrefixPool


def prepare(args):
    seed, w, models = args
    p, f = continuation(AdaptiveBeam(), w['history'], seed, particles=64)
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    truth = w['truth'][:, [49, 99, 299]]
    tasks = []
    for model in models:
        ids = np.flatnonzero(w['video'] == model['held_video'])
        tasks.append((seed, model, ids, motion[ids], p[ids][:, :, [49, 99, 299]], truth[ids], f[ids][:, :, [49, 99, 299]]))
    return tasks


def main():
    root = Path('adaptive_search_results')
    files = [root/'crossfit_output_kernel_models.json', root/'crossfit_output_kernel_evaluation.json',
             Path('v20_rnn_mixture/models/frozen_dynamics.json'), Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    frozen, old = [json.loads(p.read_text()) for p in files[:2]]
    assert frozen == old['frozen'] and frozen['all_converged']
    models = frozen['models']; engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    np.testing.assert_array_equal(w['video'], old['video']); np.testing.assert_array_equal(w['start'], old['start'])
    p, f = continuation(engine, w['history'], 541017, particles=32)
    truth = w['truth'][:, [49, 99, 299]]
    baseline = score_endpoints(p[:, :, [49, 99, 299]], truth, f[:, :, [49, 99, 299]])
    for row in old['runs']:
        if row['seed'] == 541017:
            np.testing.assert_array_equal(baseline[row['indices']], row['results']['4097']['zero'])
    model = models[0]; index = np.flatnonzero(w['video']==model['held_video'])[:1]
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    replay = evaluate_fold((541017, model, index, motion[index], p[index][:, :, [49,99,299]], truth[index], f[index][:, :, [49,99,299]]))
    reference = next(r for r in old['runs'] if r['seed']==541017 and r['held_video']==model['held_video'])
    for size in ['2049', '4097']:
        for name in ['constant', 'conditional']:
            for key in ['cost', 'gradient']:
                np.testing.assert_array_equal(replay['results'][size][name][key], reference['results'][size][name][key][:1])
    print('old all80 zero and first-window both kernels exact', flush=True)
    seeds = list(range(551017, 551021)); tasks = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for group in pool.map(prepare, [(seed, w, models) for seed in seeds]):
            tasks.extend(group)
    print('all fresh trajectories ready; evaluating frozen folds', flush=True)
    runs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(evaluate_fold, args) for args in tasks]
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('done', len(runs), 'of', len(tasks), flush=True)
    runs.sort(key=lambda r: (r['seed'], r['held_video']))
    values = {name: np.zeros((4, 80, 3)) for name in ['zero', 'constant', 'conditional']}
    for row in runs:
        si = seeds.index(row['seed']); ids = row['indices']
        for name in values:
            values[name][si, ids] = row['results']['4097']['zero'] if name=='zero' else row['results']['4097'][name]['cost']
    summary = dict(zero=float(values['zero'].mean()))
    for name in ['constant', 'conditional']:
        delta = values[name]-values['zero']; means = delta.mean((1, 2))
        summary[name] = dict(score=float(values[name].mean()), delta=float(delta.mean()), seed_deltas=means.tolist(),
                             conditional_seed_se=float(means.std(ddof=1)/np.sqrt(len(means))), better_seeds=int((means<0).sum()),
                             horizon_delta=delta.mean((0, 1)).tolist(),
                             per_video_delta={str(v): float(delta[:, w['video']==v].mean()) for v in np.unique(w['video'])})
    difference = values['conditional']-values['constant']
    summary['conditional_minus_constant'] = dict(mean=float(difference.mean()), seed_deltas=difference.mean((1,2)).tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = dict(source_hashes=hashes, runs=runs, summary=summary, frozen=frozen,
                  video=w['video'].tolist(), start=w['start'].tolist(), old_zero_exact=True, old_first_window_kernel_exact=True,
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for row in runs for g in row['gates'].values()),
                  note='Frozen all10folds constant/conditional/zero. New551017-20/P64 versus old2seed/P32:4xtrajectorybudget. '
                       'All80TRAINstates,adapterCV not backboneblindtest;seedSE fixed-window only. '
                       'All2049/4097score1e-4/grad1e-5. No refit/threshold/boundary change/hold/DEV/TEST/promotion.')
    (root/'crossfit_output_kernel_confirmation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, gates=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
