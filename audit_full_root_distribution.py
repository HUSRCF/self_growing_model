"""IID root-mixture energy scores, integrating the root categorical choices."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_value_coverage import candidate_features
from ensemble_score_objective import energy_costs
from training_window_sampler import TrainingPrefixPool


def mixture_terms(x, truth, failed):
    """x[N,R,P,D]; exclude shared RNG index i=j even ACROSS components."""
    n, r, p, d = x.shape
    if p < 3 or truth.shape != (n, d) or failed.shape != (n, r, p):
        raise ValueError('Expected >=3 particles and matching truth/failure dimensions')
    a = (np.linalg.norm(x-truth[:, None, None], axis=-1)+2*failed).mean(-1)
    b = np.empty((n, r, r)); independent = ~np.eye(p, dtype=bool)
    for i in range(r):
        for j in range(r):
            distance = np.linalg.norm(x[:, i, :, None]-x[:, j, None, :], axis=-1)
            b[:, i, j] = distance[:, independent].mean(-1)
        direct = energy_costs(x[:, i], truth, failed[:, i])[0]
        np.testing.assert_allclose(a[:, i]-.5*b[:, i, i], direct, rtol=0, atol=1e-14)
    np.testing.assert_allclose(b, b.transpose(0, 2, 1), rtol=0, atol=1e-14)
    return a, b


def mixture_score(a, b, probability):
    w = np.asarray(probability)
    if w.shape != a.shape or not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(-1), 1):
        raise ValueError('Invalid mixture probabilities')
    return (w*a).sum(-1)-.5*np.einsum('nr,nrs,ns->n', w, b, w)


def change_terms(a, b, probability, reference):
    d = probability-reference
    linear = (d*a).sum(-1)-np.einsum('nr,nrs,ns->n', d, b, reference)
    quadratic = -.5*np.einsum('nr,nrs,ns->n', d, b, d)
    return linear, quadratic


def main():
    root = Path('adaptive_search_results'); source = root/'matched_value_targets.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); old = json.loads(source.read_text())
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(401017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
    _, prior = candidate_features(engine, w['history']); policies = {'zero': prior}
    for name, model in old['models'].items():
        hard = np.eye(8)[np.array(model['scores']).argmin(1)]
        policies[name+'_hard'] = hard
        policies[name+'_soft'] = np.array(model['probability'])
        policies[name+'_mix25'] = .75*prior+.25*hard
    runs = {name: [] for name in policies}; moments = []
    for seed in range(451017, 451021):
        endpoints, failures, guards = [], [], []
        for r in range(8):
            prediction, failed = continuation(engine, w['history'], seed, root=r)
            endpoints.append(prediction[:, :, [49, 99, 299]])
            failures.append(failed[:, :, [49, 99, 299]])
            guards.append(int(failed.any(-1).sum()))
        x, dead = embedding(np.stack(endpoints, 1)), np.stack(failures, 1)
        aa, bb = [], []
        for i, t in enumerate((50, 100, 300)):
            a, b = mixture_terms(x[:, :, :, i], embedding(w['truth'][:, t-1]), dead[:, :, :, i])
            aa.append(a); bb.append(b)
        a, b = np.mean(aa, 0), np.mean(bb, 0)
        base = mixture_score(a, b, prior)
        for name, probability in policies.items():
            cost = mixture_score(a, b, probability); linear, quadratic = change_terms(a, b, probability, prior)
            np.testing.assert_allclose(cost-base, linear+quadratic, rtol=0, atol=1e-14)
            runs[name].append(dict(seed=seed, cost=float(cost.mean()), delta=float((cost-base).mean()),
                                   linear=float(linear.mean()), quadratic=float(quadratic.mean()),
                                   window_cost=cost.tolist(), per_video_delta={str(v): float((cost-base)[w['video']==v].mean()) for v in np.unique(w['video'])}))
        moments.append(dict(seed=seed, attraction=a.tolist(), pair_distance=b.tolist(), guards=guards))
        print('IID root mixture', seed, {k: v[-1]['delta'] for k, v in runs.items()}, flush=True)
    summary = {}
    for name, rows in runs.items():
        delta = np.array([r['delta'] for r in rows])
        summary[name] = dict(cost=float(np.mean([r['cost'] for r in rows])), delta=float(delta.mean()),
                             conditional_seed_se=float(delta.std(ddof=1)/2), better_seeds=int((delta < 0).sum()),
                             linear=float(np.mean([r['linear'] for r in rows])), quadratic=float(np.mean([r['quadratic'] for r in rows])),
                             per_video_delta={v: float(np.mean([r['per_video_delta'][v] for r in rows])) for v in rows[0]['per_video_delta']})
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(summary=summary, runs=runs, moments=moments, policies={k: v.tolist() for k, v in policies.items()},
                  source_sha256=digest, video=w['video'].tolist(), start=w['start'].tolist(),
                  note='All3 matched-target critics frozen,prior/hard/soft/fixed25%hard+75%prior;no selection. New451017-20/P4 percomponent/300 same80TRAINfitroots. Root choices integrated exactly,continuation MonteCarlo: J(w)=wA-.5wBw for IID draws from SAME new root distribution. B excludes same RNG particle index i=j within AND across components;common RNG across components otherwise paired. One-hot score equals original energy_costs;tested exact expectation over independent root assignments in toy. Only root changed,not repeated controller/newvideos. Finite B estimates need not imply nonnegative quadratic changes. No hold/DEV/TEST/tuning/default promotion.')
    (root/'full_root_distribution.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
