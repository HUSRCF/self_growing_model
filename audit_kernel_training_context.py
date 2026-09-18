"""TRAIN state coverage and fresh-RNG fixed-kernel labels; descriptive, not CV."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from audit_periodic_output_kernel import score_endpoints
from audit_energy_action_value import embedding
from spectral_kernel_energy import blocked_value
from v20_rnn_mixture.engine.common import DT


def score_chunk(args):
    points, truth, failed, kappa = args
    results = {}
    for size in [2049, 4097]:
        cost, gradient = [], []
        for x, y, f in zip(points, truth, failed):
            rows = [blocked_value(x[:, t], y[t], f[:, t], size, np.log(kappa[t])) for t in range(3)]
            cost.append([r['cost'] for r in rows]); gradient.append([r['gradient'].tolist() for r in rows])
        results[str(size)] = dict(cost=cost, gradient=gradient)
    return results


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    root = Path('adaptive_search_results')
    fitted = json.loads((root/'energy_output_kernel_fit.json').read_text())
    old = json.loads((root/'energy_kernel_refinement.json').read_text())
    engine = AdaptiveBeam(); pool = TrainingPrefixPool(engine.base, steps=300)
    w = pool.sample(481017, per_video=8)
    np.testing.assert_array_equal(w['video'], old['video']); np.testing.assert_array_equal(w['start'], old['start'])
    motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
    edges = np.quantile(motion, [.25, .5, .75]); bins = np.searchsorted(edges, motion, side='right')
    coverage = {str(v): dict(eligible=len(p['starts']), motion_quantiles=np.quantile(p['motion']*DT, [0, .25, .5, .75, 1]).tolist(),
                              fractions_in_fit80_bins=(np.bincount(np.searchsorted(edges, p['motion']*DT, side='right'), minlength=4)/len(p['starts'])).tolist())
                for v, p in pool.pool.items()}
    kappa = [r['kappa'] for r in fitted['fits']]
    # Replay all original baselines before generating the independent action stream.
    p, f = continuation(engine, w['history'], 482017, particles=32)
    truth = w['truth'][:, [49, 99, 299]]
    baseline = score_endpoints(p[:, :, [49, 99, 299]], truth, f[:, :, [49, 99, 299]])
    np.testing.assert_array_equal(baseline, old['results']['4097']['baseline'])
    old_delta = np.array(old['results']['4097']['cost'])-baseline
    print('old baselines exact; generating fresh TRAIN labels', flush=True)
    p, f = continuation(engine, w['history'], 532017, particles=32)
    points, failed = p[:, :, [49, 99, 299]], f[:, :, [49, 99, 299]]
    baseline = score_endpoints(points, truth, failed)
    with ProcessPoolExecutor(max_workers=4) as workers:
        chunks = list(workers.map(score_chunk, [(points[i:i+20], truth[i:i+20], failed[i:i+20], kappa) for i in range(0, 80, 20)]))
    results = {str(size): {key: np.concatenate([chunk[str(size)][key] for chunk in chunks], axis=0)
                           for key in ['cost', 'gradient']} for size in [2049, 4097]}
    errors = {key: float(np.max(np.abs(results['2049'][key]-results['4097'][key]))) for key in ['cost', 'gradient']}
    delta = results['4097']['cost']-baseline
    dispersion = np.stack([embedding(points[:, :, t]).var(axis=1).sum(-1) for t in range(3)], axis=1)
    q, memory = engine.machine.initialize(w['history'])
    pe, tr, _ = engine.machine.read(w['history'], q, memory)
    prior = np.einsum('ne,ner->nr', pe, tr)
    entropy = -(prior*np.log(np.maximum(prior, 1e-300))).sum(1)
    log_motion = np.log(motion+1e-12)
    centered_motion, centered_delta = log_motion.copy(), delta.mean(1).copy()
    for v in np.unique(w['video']):
        mask = w['video'] == v
        centered_motion[mask] -= centered_motion[mask].mean(); centered_delta[mask] -= centered_delta[mask].mean()
    diagnostic = dict(old_delta=old_delta.mean(0).tolist(), fresh_delta=delta.mean(0).tolist(),
                      old_fresh_window_correlation=corr(old_delta.mean(1), delta.mean(1)),
                      log_motion_delta_correlation=corr(log_motion, delta.mean(1)),
                      within_video_log_motion_delta_correlation=corr(centered_motion, centered_delta),
                      prior_entropy_delta_correlation=corr(entropy, delta.mean(1)),
                      horizon_dispersion_delta_correlation=[corr(dispersion[:, t], delta[:, t]) for t in range(3)],
                      motion_bins=[dict(count=int((bins==b).sum()), old_delta=float(old_delta[bins==b].mean()),
                                        fresh_delta=float(delta[bins==b].mean()), horizon_delta=delta[bins==b].mean(0).tolist()) for b in range(4)])
    report = dict(coverage=coverage, motion_edges=edges.tolist(), diagnostic=diagnostic,
                  video=w['video'].tolist(), start=w['start'].tolist(), motion=motion.tolist(), dispersion=dispersion.tolist(),
                  entropy=entropy.tolist(), old_delta=old_delta.tolist(), fresh_delta=delta.tolist(), baseline=baseline.tolist(),
                  results={size: {key: value.tolist() for key, value in row.items()} for size, row in results.items()},
                  errors=errors, cost_pass=errors['cost']<=1e-4, gradient_pass=errors['gradient']<=1e-5,
                  guards=int(f.any(-1).sum()),
                  note='TRAIN10prefix only. Same80fittingstates,old482017 replay,new532017/P32 frozen U kernel. '
                       'All80new2049/4097gates. Bins fromfit80motion only;fullpool coverage descriptive. '
                       'Feature correlations and bins not trained gates or honest CV:kernel fitted all10videos. '
                       'New RNG independence not new windows/videos. No hold/DEV/TEST/refit/promotion.')
    (root/'kernel_training_context.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(diagnostic=diagnostic, motion_edges=edges.tolist(), errors=errors), indent=2), flush=True)


if __name__ == '__main__':
    main()
