"""Adapter-level leave-one-video-out: constant versus two-bin causal width."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from training_window_sampler import TrainingPrefixPool
from fit_energy_output_kernel import mean_statistics
from spectral_kernel_energy import value_gradient, blocked_value


def fold_partition(motion, video, held):
    train = video != held
    if not train.any() or train.all():
        raise ValueError('Need both fitting and excluded videos')
    threshold = float(np.median(motion[train]))
    return train, threshold, (motion >= threshold).astype(int)


def combine_statistics(groups, counts):
    total = dict(modes=groups[0]['modes'], coeff=groups[0]['coeff'])
    for key in ['attraction', 'pair', 'baseline', 'penalty']:
        total[key] = sum(g[key]*n for g, n in zip(groups, counts))/sum(counts)
    return total


def fit_one(stats):
    def objective(theta):
        value, gradient = value_gradient(stats, theta)
        return float(value[0]), gradient[0]
    r = minimize(objective, np.log([10., 10.]), jac=True, method='L-BFGS-B',
                 bounds=[(np.log(1e-4), np.log(1e6))]*2,
                 options=dict(maxiter=200, gtol=1e-7, ftol=1e-12))
    zero = float(stats['baseline'][0])
    return dict(kappa=np.exp(r.x).tolist(), train_cost=float(r.fun), zero_cost=zero,
                use_zero=bool(r.fun >= zero), success=bool(r.success), message=str(r.message),
                iterations=int(r.nit), gradient=r.jac.tolist(),
                at_boundary=((r.x < np.log(1e-4)+1e-6)|(r.x > np.log(1e6)-1e-6)).tolist())


def fit_fold(args):
    held, motion, video, points, truth, failed = args
    train, threshold, bins = fold_partition(motion, video, held)
    constant, conditional = [], [[], []]
    counts = [int((train & (bins==b)).sum()) for b in range(2)]
    for t in range(3):
        groups = []
        for b in range(2):
            select = train & (bins == b)
            groups.append(mean_statistics(points[select, :, t], truth[select, t], failed[select, :, t], 1025))
            conditional[b].append(fit_one(groups[-1]))
        constant.append(fit_one(combine_statistics(groups, counts)))
    return dict(held_video=int(held), fit_videos=np.unique(video[train]).tolist(), threshold=threshold,
                counts=counts, constant=constant, conditional=conditional)


def evaluate_fold(args):
    seed, model, indices, motion, points, truth, failed = args
    bins = (motion >= model['threshold']).astype(int)
    results = {}
    for size in [2049, 4097]:
        data = {name: dict(cost=[], gradient=[]) for name in ['constant', 'conditional']}
        zero = []
        for j in range(len(points)):
            z = []
            for name in data:
                values, gradients = [], []
                specs = model['constant'] if name == 'constant' else model['conditional'][bins[j]]
                for t, spec in enumerate(specs):
                    theta = None if spec['use_zero'] else np.log(spec['kappa'])
                    row = blocked_value(points[j, :, t], truth[j, t], failed[j, :, t], size, theta)
                    values.append(row['cost']); gradients.append(row['gradient'].tolist())
                    if name == 'constant':
                        z.append(row['baseline'])
                data[name]['cost'].append(values); data[name]['gradient'].append(gradients)
            zero.append(z)
        results[str(size)] = dict(**data, zero=zero)
    gates = {}
    for name in ['constant', 'conditional']:
        errors = {key: float(np.max(np.abs(np.array(results['2049'][name][key])-results['4097'][name][key])))
                  for key in ['cost', 'gradient']}
        gates[name] = dict(**errors, cost_pass=errors['cost'] <= 1e-4, gradient_pass=errors['gradient'] <= 1e-5)
    return dict(seed=seed, held_video=model['held_video'], indices=indices.tolist(), results=results,
                gates=gates, guards=int(failed.sum()))


def main():
    root = Path('adaptive_search_results')
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    p, f = continuation(engine, w['history'], 482017, particles=32)
    points, truth, failed = p[:, :, [49, 99, 299]], w['truth'][:, [49, 99, 299]], f[:, :, [49, 99, 299]]
    old = json.loads((root/'energy_kernel_refinement.json').read_text())
    np.testing.assert_array_equal(w['video'], old['video']); np.testing.assert_array_equal(w['start'], old['start'])
    np.testing.assert_array_equal(score_endpoints(points, truth, failed), old['results']['4097']['baseline'])
    if failed.any():
        raise RuntimeError('Do not fit with failed trajectories')
    models = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fit_fold, (v, motion, w['video'], points, truth, failed)) for v in np.unique(w['video'])]
        for job in as_completed(jobs):
            model = job.result(); models.append(model)
            print('fit fold', model['held_video'], flush=True)
    models.sort(key=lambda m: m['held_video'])
    fits = [fit for model in models for group in [model['constant'], *model['conditional']] for fit in group]
    frozen = dict(models=models, all_converged=all(fit['success'] for fit in fits),
                  zero_choices=sum(fit['use_zero'] for fit in fits), boundary_fits=sum(any(fit['at_boundary']) for fit in fits),
                  note='10folds,adapter refit9videos,foldmedian historymotion,2bin conditional12widthparams vsconstant6; '
                       '1025U/singlelog10start/bounds1e-4..1e6/maxiter200. TRAINmean choosezero if fit not better. '
                       'Backbone trained all TRAIN: adapter-level crossfit only. Frozen before new evaluation RNG.')
    (root/'crossfit_output_kernel_models.json').write_text(json.dumps(frozen, indent=2))
    if not frozen['all_converged']:
        print('Nonconverged fits preserved; evaluation not started', flush=True)
        return
    tasks = []
    for seed in [541017, 541018]:
        p, f = continuation(engine, w['history'], seed, particles=32)
        for model in models:
            ids = np.flatnonzero(w['video'] == model['held_video'])
            tasks.append((seed, model, ids, motion[ids], p[ids][:, :, [49, 99, 299]], truth[ids], f[ids][:, :, [49, 99, 299]]))
    runs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(evaluate_fold, args) for args in tasks]
        for job in as_completed(jobs):
            row = job.result(); runs.append(row)
            print('eval', row['seed'], row['held_video'], flush=True)
    runs.sort(key=lambda r: (r['seed'], r['held_video']))
    values = {name: np.zeros((2, 80, 3)) for name in ['zero', 'constant', 'conditional']}
    for row in runs:
        si = [541017, 541018].index(row['seed']); ids = row['indices']
        for name in values:
            values[name][si, ids] = row['results']['4097']['zero'] if name=='zero' else row['results']['4097'][name]['cost']
    summary = dict(zero=float(values['zero'].mean()))
    for name in ['constant', 'conditional']:
        delta = values[name]-values['zero']
        summary[name] = dict(score=float(values[name].mean()), delta=float(delta.mean()),
                             seed_deltas=delta.mean((1, 2)).tolist(), horizon_delta=delta.mean((0, 1)).tolist(),
                             per_video_delta={str(v): float(delta[:, w['video']==v].mean()) for v in np.unique(w['video'])})
    summary['conditional_minus_constant'] = float((values['conditional']-values['constant']).mean())
    report = dict(frozen=frozen, runs=runs, summary=summary, video=w['video'].tolist(), start=w['start'].tolist(),
                  all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for row in runs for g in row['gates'].values()),
                  note='Two fresh independent action streams after freezing allfolds,all80statesP32. '
                       'All2049/4097score1e-4/grad1e-5. No hold/DEV/TEST or model promotion. '
                       'State labels/thresholds excluded per video,not backbone-level blind validation; '
                       'training optimization path not precision-audited,final evaluation only.')
    (root/'crossfit_output_kernel_evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, gates=report['all_precision_gates_pass']), indent=2), flush=True)


if __name__ == '__main__':
    main()
