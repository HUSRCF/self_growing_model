"""Numerical gates only: TRAIN roots, fixed kernels, no new fitting or holdout."""
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from periodic_output_kernel import convolve, nll_gradient
from audit_periodic_output_kernel import score_endpoints
from spectral_kernel_energy import statistics, value_gradient, spectral_cost, kernel_moments


def main():
    root = Path('adaptive_search_results'); old = json.loads((root/'periodic_output_kernel.json').read_text())
    output = root/'spectral_kernel_energy_audit.json'
    previous = json.loads(output.read_text()) if output.exists() else None
    newer = json.loads((root/'kernel_particle_coverage.json').read_text())
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    p, f = continuation(engine, w['history'], 482017, particles=32)
    points = p[:, :, [49, 99, 299]]; truth = w['truth'][:, [49, 99, 299]]; dead = f[:, :, [49, 99, 299]]
    for t in range(3):
        assert nll_gradient(np.log(old['kappa'][t]), points[:, :, t], truth[:, t])[0] == old['fits'][t]['nll']
    selected = np.arange(0, 80, 8)
    points, truth, dead = points[selected], truth[selected], dead[selected]
    kernels = {k: np.array(v) for k, v in newer['kernels'].items()}; results = {}; timings = {}; baseline_errors = {}
    for size in [33, 65, 129, 257]:
        start = time.process_time(); stats = [statistics(points[:, :, t], truth[:, t], dead[:, :, t], size) for t in range(3)]
        timings[str(size)] = dict(precompute_cpu_seconds=time.process_time()-start)
        baseline_errors[str(size)] = np.stack([spectral_cost(s, np.ones((size, size)))-s['baseline'] for s in stats], 1).tolist()
        results[str(size)] = {}; start = time.process_time()
        for name, kappa in kernels.items():
            values, gradients, raw_values = [], [], []
            for t in range(3):
                value, gradient = value_gradient(stats[t], np.log(kappa[t]))
                np.testing.assert_array_equal(value_gradient(stats[t])[0], stats[t]['baseline'])
                values.append(value); gradients.append(gradient)
                raw_values.append(spectral_cost(stats[t], kernel_moments(stats[t]['modes'], np.log(kappa[t]))[0]))
            results[str(size)][name] = dict(cost=np.stack(values, 1).tolist(), gradient=np.stack(gradients, 1).tolist(), raw_cost=np.stack(raw_values, 1).tolist())
            np.testing.assert_allclose(np.stack(values, 1)-np.stack(raw_values, 1), -np.array(baseline_errors[str(size)]), atol=1e-14)
            if previous is not None:
                for key in ['cost', 'gradient']:
                    np.testing.assert_array_equal(results[str(size)][name][key], previous['results'][str(size)][name][key])
        timings[str(size)]['two_kernel_eval_cpu_seconds'] = time.process_time()-start
        print('grid', size, 'complete', flush=True)
    gates = {}
    for name in kernels:
        a, b = results['129'][name], results['257'][name]
        cost_error = float(np.max(np.abs(np.array(a['cost'])-np.array(b['cost']))))
        gradient_error = float(np.max(np.abs(np.array(a['gradient'])-np.array(b['gradient']))))
        gates[name] = dict(max_cost_difference=cost_error, max_gradient_difference=gradient_error,
                           max_raw_cost_difference=float(np.max(np.abs(np.array(a['raw_cost'])-np.array(b['raw_cost'])))),
                           cost_pass=cost_error <= 1e-4, gradient_pass=gradient_error <= 1e-5)
    # Independent noise draws at fixed base points; this audits kernel integration only.
    start = time.process_time(); draws = []
    for seed in range(501017, 501529):
        draws.append(score_endpoints(convolve(points, kernels['new128'], seed), truth, dead))
    draws = np.array(draws); mean = draws.mean(0); se = draws.std(0, ddof=1)/np.sqrt(len(draws))
    value = np.array(results['257']['new128']['cost'])
    aggregate_draws = draws.mean(1)
    mc = dict(mean=mean.tolist(), conditional_noise_se=se.tolist(), max_abs_difference=float(np.max(np.abs(mean-value))),
              horizon_difference=(value.mean(0)-mean.mean(0)).tolist(),
              horizon_noise_se=(aggregate_draws.std(0, ddof=1)/np.sqrt(len(draws))).tolist(), cpu_seconds=time.process_time()-start)
    report = dict(results=results, timings=timings, gates=gates, monte_carlo=mc, baseline_spectral_errors=baseline_errors,
                  previous_cost_gradient_exact=previous is not None,
                  video=w['video'][selected].tolist(), start=w['start'][selected].tolist(), guards=int(dead.any(-1).sum()),
                  note='FixedTRAIN10 roots (first of each original8),old482017/P32 baseline,NLL exact replay. Two frozen kernels. Odd Fourier grids33/65/129/257;predeclared129vs257 maxscore1e-4/maxgradient1e-5 gates,record failures without changing limits. Zero-anchored spectral change approximation,not exact integral; finite-grid errors remain. U pair characteristic=(P|phi|²-1)/(P-1),independent kernel noises give squared multiplier. Bessel derivative w.r.t logk. 512independentnoise draws onlynew128 fixedbase,noiseSE not dynamical/video uncertainty. No fitting/hold/DEV/TEST/promotion.')
    (root/'spectral_kernel_energy_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(gates=gates, monte_carlo={k: v for k, v in mc.items() if k not in ['mean', 'conditional_noise_se']}), indent=2), flush=True)


if __name__ == '__main__':
    main()
