"""Higher-precision independent RNG check of frozen TRAIN-hold root policies."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_value_coverage import candidate_features
from audit_soft_value import tilt, read_probabilities
from audit_full_root_distribution import mixture_terms, mixture_score, change_terms
from validate_root_mixture_holdout import predict
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def collect_moments(engine, windows, seed, particles):
    endpoints, failures, guards = [], [], []
    for r in range(8):
        p, f = continuation(engine, windows['history'], seed, particles=particles, root=r)
        endpoints.append(p[:, :, [49, 99, 299]]); failures.append(f[:, :, [49, 99, 299]])
        guards.append(int(f.any(-1).sum()))
    x, dead = embedding(np.stack(endpoints, 1)), np.stack(failures, 1)
    terms = [mixture_terms(x[:, :, :, i], embedding(windows['truth'][:, t-1]), dead[:, :, :, i])
             for i, t in enumerate((50, 100, 300))]
    return np.mean([v[0] for v in terms], 0), np.mean([v[1] for v in terms], 0), terms, guards


def summarize_runs(rows):
    if len(rows) < 2:
        raise ValueError('At least two RNG runs required')
    delta = np.array([r['delta'] for r in rows])
    return dict(cost=float(np.mean([r['cost'] for r in rows])), delta=float(delta.mean()),
                conditional_seed_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                better_seeds=int((delta < 0).sum()), seed_delta=delta.tolist(),
                per_video_delta={v: float(np.mean([r['per_video_delta'][v] for r in rows])) for v in rows[0]['per_video_delta']},
                horizon_costs=np.mean([r['horizon_costs'] for r in rows], 0).tolist())


def main():
    root = Path('adaptive_search_results'); source = root/'root_mixture_holdout.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); old = json.loads(source.read_text())
    engine = AdaptiveBeam(); w = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
    x, prior = candidate_features(engine, w['history']); pe, tr = read_probabilities(engine, w['history'])
    policies = {'zero': prior}
    for name, arrays in old['models'].items():
        model = {k: np.array(v) for k, v in arrays.items()}; scores = predict(model, x)
        np.testing.assert_array_equal(scores, old['scores'][name])
        hard = np.eye(8)[scores.argmin(1)]
        policies[name+'_hard'] = hard; policies[name+'_mix25'] = .75*prior+.25*hard
        policies[name+'_soft'] = np.einsum('be,ber->br', pe, tilt(tr, scores, old['calibration'][name]['strength']))
    for name, probability in policies.items():
        np.testing.assert_array_equal(probability, old['policies'][name])
    # Re-score all old saved components, and regenerate the first old RNG/P4.
    for j, moment in enumerate(old['moments']):
        a, b = np.array(moment['attraction']), np.array(moment['pair_distance'])
        for name, probability in policies.items():
            np.testing.assert_array_equal(mixture_score(a, b, probability), old['runs'][name][j]['window_cost'])
    a, b, _, guards = collect_moments(engine, w, 461017, 4)
    np.testing.assert_array_equal(a, old['moments'][0]['attraction'])
    np.testing.assert_array_equal(b, old['moments'][0]['pair_distance'])
    assert guards == old['moments'][0]['guards']
    print('saved predictions/policies/all scores and first old P4 moments exact', flush=True)
    runs = {name: [] for name in policies}; moments = []
    for seed in range(471017, 471025):
        a, b, terms, guards = collect_moments(engine, w, seed, 8)
        baseline = mixture_score(a, b, prior)
        for name, probability in policies.items():
            cost = mixture_score(a, b, probability); linear, quadratic = change_terms(a, b, probability, prior)
            np.testing.assert_allclose(cost-baseline, linear+quadratic, rtol=0, atol=1e-14)
            runs[name].append(dict(seed=seed, cost=float(cost.mean()), delta=float((cost-baseline).mean()),
                                   window_cost=cost.tolist(), linear=float(linear.mean()), quadratic=float(quadratic.mean()),
                                   horizon_costs=[float(mixture_score(ta, tb, probability).mean()) for ta, tb in terms],
                                   per_video_delta={str(v): float((cost-baseline)[w['video']==v].mean()) for v in np.unique(w['video'])}))
        moments.append(dict(seed=seed, attraction=a.tolist(), pair_distance=b.tolist(), guards=guards))
        print('confirmation P8', seed, {k: v[-1]['delta'] for k, v in runs.items() if 'mix25' in k}, flush=True)
    summary = {name: summarize_runs(rows) for name, rows in runs.items()}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(summary=summary, runs=runs, moments=moments, source_sha256=digest,
                  video=w['video'].tolist(), start=w['start'].tolist(), exact_source_predictions_and_policies=True,
                  exact_all_saved_scores=True, exact_first_old_moments=True,
                  note='All10 frozen policies from root_mixture_holdout,NO fitting/calibration/selection. Same TRAINhold24windows;8new471017-24/P8percomponent/300 (previous4RNG/P4),increased evaluation effort not altered distribution or equal-budget comparison. Same independent-index crosscomponent pair estimator. Old predictions/policies and 40saved scores exact,first461017P4moments regenerated exact. Conditional RNG SE,not newvideos. No ratio/target tuning/DEV/TEST/default promotion.')
    (root/'root_mixture_holdout_confirmation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
