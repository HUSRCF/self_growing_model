"""Frozen TRAIN roots: soft mixture algebra, replay, and uniform alpha grid audit."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from training_window_sampler import TrainingPrefixPool
from spectral_kernel_energy import blocked_value
from soft_output_mixture import value_gradient, interval_grid_difference, exact_point_terms


def evaluate_chunk(args):
    points, truth, failed, kappa = args
    results = {}
    for size in [2049, 4097]:
        data = {key: np.zeros((len(points), 3)) for key in ['baseline', 'linear', 'quadratic', 'cost', 'exact_linear', 'exact_quadratic', 'raw_cost']}
        for j in range(len(points)):
            for t in range(3):
                row = blocked_value(points[j, :, t], truth[j, t], failed[j, :, t], size, np.log(kappa[t]), mixture=True)
                for key in data:
                    data[key][j, t] = row[key]
                np.testing.assert_allclose(value_gradient(row, 1)[0], row['cost'], atol=1e-13, rtol=0)
                assert value_gradient(row, 0)[0] == row['baseline']
                np.testing.assert_allclose(value_gradient(exact_point_terms(row),1)[0], row['raw_cost'], atol=1e-13, rtol=0)
        results[str(size)] = data
    return results


def main():
    root = Path('adaptive_search_results')
    output = root/'soft_output_mixture_audit.json'
    previous = json.loads(output.read_text()) if output.exists() else None
    old = json.loads((root/'energy_kernel_refinement.json').read_text())
    fitted = json.loads((root/'energy_output_kernel_fit.json').read_text())
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    np.testing.assert_array_equal(w['video'], old['video']); np.testing.assert_array_equal(w['start'], old['start'])
    p, f = continuation(engine, w['history'], 482017, particles=32)
    points, truth, failed = p[:, :, [49,99,299]], w['truth'][:, [49,99,299]], f[:, :, [49,99,299]]
    kappa = [row['kappa'] for row in fitted['fits']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(evaluate_chunk, [(points[i:i+20], truth[i:i+20], failed[i:i+20], kappa) for i in range(0,80,20)]))
    results = {str(size): {key: np.concatenate([chunk[str(size)][key] for chunk in chunks])
                           for key in ['baseline','linear','quadratic','cost','exact_linear','exact_quadratic','raw_cost']} for size in [2049,4097]}
    for size in ['2049','4097']:
        np.testing.assert_array_equal(results[size]['cost'], old['results'][size]['cost'])
        np.testing.assert_array_equal(results[size]['baseline'], old['results'][size]['baseline'])
        if previous is not None:
            for key in ['baseline','linear','quadratic','cost']:
                np.testing.assert_array_equal(results[size][key], previous['results'][size][key])
    bounds = interval_grid_difference(results['2049'], results['4097'])
    gates = dict(max_uniform_cost_difference=float(bounds['cost'].max()), max_uniform_alpha_gradient_difference=float(bounds['gradient'].max()),
                 cost_pass=bool(bounds['cost'].max()<=1e-4), gradient_pass=bool(bounds['gradient'].max()<=1e-5))
    corrected_bounds = interval_grid_difference(exact_point_terms(results['2049']), exact_point_terms(results['4097']))
    corrected_gates = dict(max_uniform_cost_difference=float(corrected_bounds['cost'].max()),
                          max_uniform_alpha_gradient_difference=float(corrected_bounds['gradient'].max()),
                          cost_pass=bool(corrected_bounds['cost'].max()<=1e-4), gradient_pass=bool(corrected_bounds['gradient'].max()<=1e-5))
    diagnostics = {str(alpha): dict(mean_change=float((value_gradient(results['4097'],alpha)[0]-results['4097']['baseline']).mean()),
                                   horizon_change=(value_gradient(results['4097'],alpha)[0]-results['4097']['baseline']).mean(0).tolist())
                   for alpha in [0., .25, .5, 1.]}
    report = dict(results={size:{key:value.tolist() for key,value in row.items()} for size,row in results.items()},
                  gates=gates, exact_point_gates=corrected_gates, previous_anchored_terms_exact=previous is not None,
                  diagnostics=diagnostics, video=w['video'].tolist(), start=w['start'].tolist(),
                  negative_quadratic_count=int((results['4097']['quadratic']<0).sum()), guards=int(failed.sum()),
                  original_endpoints_exact=True,
                  note='Frozen fullTRAIN U kernel,old80fitroots/482017P32. J(alpha)=J0+alphaL+alpha²Q, '
                       'independent perparticle point/kernel choices;not linear interpolation of whole ensemble scores. '
                       'Allalpha[0,1] exact max polynomial grid change,score1e-4/alpha-gradient1e-5;not absoluteerror bound. '
                       'FiniteP U quadratic need not be convex. Four fixed alpha diagnostics not tuning/selection. '
                       'Exact-point variant corrects unsmoothed attraction/pair analytically;alpha1 equals rawsmoothed,not oldanchored. '
                       'No gate fitting/hold/DEV/TEST or promotion;oldhold precision failure untouched.')
    (root/'soft_output_mixture_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=gates, exact_point_gates=corrected_gates, diagnostics=diagnostics, negative_quadratic_count=report['negative_quadratic_count']),indent=2),flush=True)


if __name__ == '__main__':
    main()
