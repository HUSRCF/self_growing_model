"""Frozen energy-fitted kernels: complete TRAIN resolution/anchor audit."""
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from spectral_kernel_energy import blocked_value


def main():
    root = Path('adaptive_search_results')
    old = json.loads((root/'energy_output_kernel_fit.json').read_text())
    engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    np.testing.assert_array_equal(w['video'], old['video'])
    np.testing.assert_array_equal(w['start'], old['start'])
    p, f = continuation(engine, w['history'], 482017, particles=32)
    results, timings = {}, {}
    for size in [2049, 4097]:
        start = time.process_time()
        data = {key: np.zeros((80, 3)) for key in ['cost', 'raw_cost', 'baseline_error', 'baseline']}
        data['gradient'] = np.zeros((80, 3, 2))
        for j in range(80):
            for t, step in enumerate([49, 99, 299]):
                result = blocked_value(p[j, :, step], w['truth'][j, step], f[j, :, step],
                                       size, np.log(old['fits'][t]['kappa']))
                for key in data:
                    data[key][j, t] = result[key]
            if (j+1) % 20 == 0:
                print('grid', size, 'windows', j+1, flush=True)
        if size == 2049:
            np.testing.assert_allclose(data['cost'], old['values']['2049'], atol=1e-13, rtol=0)
            np.testing.assert_allclose(data['gradient'], old['gradients']['2049'], atol=1e-13, rtol=0)
        results[str(size)] = {key: value.tolist() for key, value in data.items()}
        timings[str(size)] = time.process_time()-start
    differences = {key: float(np.max(np.abs(np.array(results['2049'][key])-np.array(results['4097'][key]))))
                   for key in ['cost', 'raw_cost', 'baseline_error', 'gradient']}
    gates = dict(cost_pass=differences['cost'] <= 1e-4, gradient_pass=differences['gradient'] <= 1e-5)
    report = dict(results=results, differences=differences, gates=gates, cpu_seconds=timings,
                  original_failed_gates=old['gates'], video=old['video'], start=old['start'],
                  guards=int(f[:, :, [49, 99, 299]].sum()),
                  note='All80TRAIN,unchanged frozen U fits,2049 replay atol1e-13 rtol0. '
                       'Full frequency row blocks128,no truncation.2049vs4097 same score1e-4/grad1e-5. '
                       'No refit/hold/DEV/TEST/promotion;grid convergence not rigorous absolute-error bound.')
    (root/'energy_kernel_refinement.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(differences=differences, gates=gates, cpu_seconds=timings), indent=2), flush=True)


if __name__ == '__main__':
    main()
