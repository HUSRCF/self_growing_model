"""TRAIN-only energy-score kernel pilot; frozen centers and independent resolution audit."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from periodic_output_kernel import nll_gradient
from spectral_kernel_energy import statistics, value_gradient


def mean_statistics(centers, truth, failed, size):
    """Exact linear aggregation of per-window spectral sufficient statistics."""
    total = None
    for j in range(len(centers)):
        s = statistics(centers[j:j+1], truth[j:j+1], failed[j:j+1], size)
        if total is None:
            total = s
        else:
            for key in ['attraction', 'pair', 'baseline', 'penalty']:
                total[key] += s[key]
    for key in ['attraction', 'pair', 'baseline', 'penalty']:
        total[key] /= len(centers)
    return total


def main():
    root = Path('adaptive_search_results')
    old = json.loads((root/'periodic_output_kernel.json').read_text())
    engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    p, f = continuation(engine, w['history'], 482017, particles=32)
    points, truth, failed = p[:, :, [49, 99, 299]], w['truth'][:, [49, 99, 299]], f[:, :, [49, 99, 299]]
    if failed.any():
        raise RuntimeError('Failed trajectories: do not silently remove windows')
    for t in range(3):
        assert nll_gradient(np.log(old['kappa'][t]), points[:, :, t], truth[:, t])[0] == old['fits'][t]['nll']
    fits = []
    start = time.process_time()
    for t in range(3):
        s = mean_statistics(points[:, :, t], truth[:, t], failed[:, :, t], 1025)
        def objective(theta):
            value, gradient = value_gradient(s, theta)
            return float(value[0]), gradient[0]
        result = minimize(objective, np.log([10., 10.]), jac=True, method='L-BFGS-B',
                          bounds=[(np.log(1e-4), np.log(1e6))]*2,
                          options=dict(maxiter=200, gtol=1e-7, ftol=1e-12))
        fits.append(dict(kappa=np.exp(result.x).tolist(), success=bool(result.success),
                         message=str(result.message), iterations=int(result.nit), gradient=result.jac.tolist(),
                         train_cost=float(result.fun), zero_cost=float(s['baseline'][0]),
                         at_boundary=((result.x < np.log(1e-4)+1e-6)|(result.x > np.log(1e6)-1e-6)).tolist()))
        print('fit', t, fits[-1], flush=True)
        del s
    fitting_cpu = time.process_time()-start
    # Freeze all fits before refinement; do not refit on the finer-grid result.
    values, gradients = {}, {}
    start = time.process_time()
    for size in [1025, 2049]:
        v, g = np.zeros((80, 3)), np.zeros((80, 3, 2))
        for j in range(80):
            for t in range(3):
                s = statistics(points[j:j+1, :, t], truth[j:j+1, t], failed[j:j+1, :, t], size)
                value, grad = value_gradient(s, np.log(fits[t]['kappa']))
                v[j, t], g[j, t] = value[0], grad[0]
                del s
            if (j+1) % 20 == 0:
                print('audit', size, j+1, flush=True)
        values[str(size)], gradients[str(size)] = v.tolist(), g.tolist()
    np.testing.assert_allclose(np.array(values['1025']).mean(0), [x['train_cost'] for x in fits], atol=1e-13, rtol=0)
    error = np.abs(np.array(values['1025'])-np.array(values['2049']))
    grad_error = np.abs(np.array(gradients['1025'])-np.array(gradients['2049']))
    gates = dict(max_cost_difference=float(error.max()), max_gradient_difference=float(grad_error.max()),
                 cost_pass=bool(error.max() <= 1e-4), gradient_pass=bool(grad_error.max() <= 1e-5))
    report = dict(fits=fits, values=values, gradients=gradients, gates=gates,
                  refined_train_delta=(np.array(values['2049']).mean(0)-np.array([x['zero_cost'] for x in fits])).tolist(),
                  fitting_cpu_seconds=fitting_cpu, audit_cpu_seconds=time.process_time()-start,
                  video=w['video'].tolist(), start=w['start'].tolist(), guards=int(failed.sum()),
                  note='TRAIN80/P32 same481017 windows/482017 dynamics. Three independent 2D widths, '
                       '1025 anchored U mean, one log10 start, bounds1e-4..1e6,maxiter200. '
                       'Frozen fit audit all80 at1025/2049,score1e-4/gradient1e-5. No fit selection, '
                       'hold/DEV/TEST or promotion. Train improvement is not generalization evidence.')
    (root/'energy_output_kernel_fit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(gates=gates, delta=report['refined_train_delta']), indent=2), flush=True)


if __name__ == '__main__':
    main()
