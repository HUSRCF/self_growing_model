"""Fixed TRAIN cases: resolution refinement, without fitting or holdout access."""
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from spectral_kernel_energy import statistics, value_gradient, spectral_cost, kernel_moments


def main():
    root = Path('adaptive_search_results')
    previous = json.loads((root/'spectral_kernel_energy_audit.json').read_text())
    kernels = json.loads((root/'kernel_particle_coverage.json').read_text())['kernels']
    engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    p, f = continuation(engine, w['history'], 482017, particles=32)
    selected = np.arange(0, 80, 8)
    np.testing.assert_array_equal(w['video'][selected], previous['video'])
    np.testing.assert_array_equal(w['start'][selected], previous['start'])
    results, timings = {}, {}
    for size in [257, 513, 1025, 2049]:
        start = time.process_time()
        out = {name: dict(cost=np.zeros((10, 3)), raw_cost=np.zeros((10, 3)),
                          gradient=np.zeros((10, 3, 2))) for name in kernels}
        errors = np.zeros((10, 3))
        # One window/horizon at a time bounds the large complex-array working set.
        for j, index in enumerate(selected):
            for t, step in enumerate([49, 99, 299]):
                s = statistics(p[index:index+1, :, step], w['truth'][index:index+1, step],
                               f[index:index+1, :, step], size)
                errors[j, t] = spectral_cost(s, np.ones((size, size)))[0]-s['baseline'][0]
                np.testing.assert_array_equal(value_gradient(s)[0], s['baseline'])
                for name, kappa in kernels.items():
                    logk = np.log(kappa[t])
                    value, gradient = value_gradient(s, logk)
                    raw = spectral_cost(s, kernel_moments(s['modes'], logk)[0])
                    out[name]['cost'][j, t] = value[0]
                    out[name]['raw_cost'][j, t] = raw[0]
                    out[name]['gradient'][j, t] = gradient[0]
                    np.testing.assert_allclose(value-raw, -errors[j, t], atol=1e-14)
                del s
        results[str(size)] = {name: {key: val.tolist() for key, val in data.items()}
                              for name, data in out.items()}
        results[str(size)]['baseline_error'] = errors.tolist()
        if size == 257:
            for name in kernels:
                for key in ['cost', 'raw_cost', 'gradient']:
                    # Chunking changes contraction order; require float64 roundoff agreement.
                    np.testing.assert_allclose(out[name][key], previous['results']['257'][name][key],
                                               atol=1e-13, rtol=0)
        timings[str(size)] = time.process_time()-start
        print('grid', size, 'cpu_seconds', timings[str(size)], flush=True)
    comparisons = {}
    for a, b in [(257, 513), (513, 1025), (1025, 2049)]:
        comparisons[f'{a}-{b}'] = {}
        for name in kernels:
            diffs = {key: float(np.max(np.abs(np.array(results[str(a)][name][key])-
                                             np.array(results[str(b)][name][key]))))
                     for key in ['cost', 'raw_cost', 'gradient']}
            diffs.update(cost_pass=diffs['cost'] <= 1e-4, raw_cost_pass=diffs['raw_cost'] <= 1e-4,
                         gradient_pass=diffs['gradient'] <= 1e-5)
            comparisons[f'{a}-{b}'][name] = diffs
    report = dict(results=results, cpu_seconds=timings, comparisons=comparisons,
                  original_failed_gates=previous['gates'], video=previous['video'], start=previous['start'],
                  guards=int(f[selected][:, :, [49, 99, 299]].any(-1).sum()),
                  note='Same fixed TRAIN10 cases and frozen kernels. Chunk1;257 replay atol1e-13 rtol0. '
                       'Unchanged score1e-4/gradient1e-5 limits. Refinement differences are not rigorous '
                       'absolute-error bounds. No fitting/hold/DEV/TEST or promotion.')
    (root/'spectral_refinement.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
