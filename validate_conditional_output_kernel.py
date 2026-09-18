"""Full TRAIN refit of fixed two-bin specification, frozen prefix/tail evaluation."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from crossfit_output_kernels import fit_one, evaluate_fold
from fit_energy_output_kernel import mean_statistics
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import tail_windows


def evaluate_region(seed, region, model):
    if region == 'prefix':
        w = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    elif region == 'tail':
        full = tail_windows('train', 300, 8)
        mask = np.isin(full['video'], SPLITS['train'][-3:])
        w = {key: value[mask] for key, value in full.items()}
    else:
        raise ValueError('Unknown region')
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    p, f = continuation(AdaptiveBeam(), w['history'], seed, particles=64)
    row = evaluate_fold((seed, model, np.arange(24), motion, p[:, :, [49, 99, 299]],
                         w['truth'][:, [49, 99, 299]], f[:, :, [49, 99, 299]]))
    row.pop('held_video')  # Full-fit model sentinel is not an excluded fold.
    row.update(region=region, video=w['video'].tolist(), start=w['start'].tolist(), motion=motion.tolist(),
               bin_counts=np.bincount((motion>=model['threshold']).astype(int), minlength=2).tolist())
    return row


def main():
    root = Path('adaptive_search_results')
    paths = [root/'energy_output_kernel_fit.json', root/'crossfit_output_kernel_models.json',
             Path('v20_rnn_mixture/models/frozen_dynamics.json'), Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    constant_source = json.loads(paths[0].read_text())
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    p, f = continuation(engine, w['history'], 482017, particles=32)
    points, truth, failed = p[:, :, [49, 99, 299]], w['truth'][:, [49, 99, 299]], f[:, :, [49, 99, 299]]
    if failed.any():
        raise RuntimeError('Fit guard; no sample exclusion')
    old = json.loads((root/'energy_kernel_refinement.json').read_text())
    np.testing.assert_array_equal(w['video'], old['video']); np.testing.assert_array_equal(w['start'], old['start'])
    np.testing.assert_array_equal(score_endpoints(points, truth, failed), old['results']['4097']['baseline'])
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    threshold = float(np.median(motion)); bins = (motion >= threshold).astype(int)
    conditional = [[], []]
    for b in range(2):
        select = bins == b
        for t in range(3):
            stats = mean_statistics(points[select, :, t], truth[select, t], failed[select, :, t], 1025)
            result = fit_one(stats); conditional[b].append(result)
            print('fit', b, t, result, flush=True)
    constant = [dict(row, use_zero=row['train_cost']>=row['zero_cost']) for row in constant_source['fits']]
    model = dict(held_video=-1, threshold=threshold, counts=np.bincount(bins, minlength=2).tolist(),
                 constant=constant, conditional=conditional, fit_videos=np.unique(w['video']).tolist())
    fits = [row for group in conditional for row in group]
    frozen = dict(model=model, source_hashes=hashes, all_converged=all(row['success'] for row in fits),
                  note='Same fixed CV specification fullTRAIN10/80/P32,trainmedian threshold;constant original allTRAIN Ufit. '
                       'held_video=-1 is evaluator interface sentinel,not a fold. Frozen before loading holdout regions.')
    model_path = root/'conditional_output_kernel_model.json'
    model_path.write_text(json.dumps(frozen, indent=2))
    hashes[str(model_path)] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not frozen['all_converged']:
        print('Nonconvergence preserved; no holdout evaluation', flush=True)
        return
    runs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(evaluate_region, seed, region, model) for seed in range(561017, 561021) for region in ['prefix', 'tail']]
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('done', row['region'], row['seed'], row['gates'], flush=True)
    runs.sort(key=lambda row: (row['region'], row['seed']))
    summary = {}
    for region in ['prefix', 'tail']:
        selected = [row for row in runs if row['region']==region]
        video = np.array(selected[0]['video'])
        zero = np.array([row['results']['4097']['zero'] for row in selected])
        result = dict(zero=float(zero.mean()), bin_counts=selected[0]['bin_counts'])
        for name in ['constant', 'conditional']:
            value = np.array([row['results']['4097'][name]['cost'] for row in selected])
            delta = value-zero; means = delta.mean((1, 2))
            result[name] = dict(score=float(value.mean()), delta=float(delta.mean()), seed_deltas=means.tolist(),
                                conditional_seed_se=float(means.std(ddof=1)/np.sqrt(len(means))), better_seeds=int((means<0).sum()),
                                horizon_delta=delta.mean((0, 1)).tolist(),
                                per_video_delta={str(v): float(delta[:, video==v].mean()) for v in np.unique(video)})
        result['conditional_minus_constant'] = result['conditional']['score']-result['constant']['score']
        summary[region] = result
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = dict(frozen=frozen, source_hashes=hashes, runs=runs, summary=summary,
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for row in runs for g in row['gates'].values()),
                  note='Frozen fullfit conditional/constant/zero,hold18/16/13prefix ANDtail24each/new561017-20/P64. '
                       'Allwindow2049/4097score1e-4/gradient1e-5;8workersBLAS1. Historically reusedhold/backboneTRAIN; '
                       'seedSE conditionalfixedwindows not videoCI. No threshold/width selection/DEV/TEST/promotion.')
    (root/'conditional_output_kernel_holdout.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, gates=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
