"""Future-label oracle diagnostic, never a deployable controller."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_full_root_distribution import mixture_score


def candidate_costs(a, b, prior):
    """Fixed candidates: unchanged prior and eight 25%-toward-root mixtures."""
    policies = np.stack([prior] + [.75*prior + .25*np.broadcast_to(row, prior.shape)
                                  for row in np.eye(prior.shape[1])], axis=1)
    cost = np.stack([mixture_score(a, b, policies[:, i])
                     for i in range(policies.shape[1])], axis=1)
    return cost, policies


def cross_stream(cost):
    if cost.ndim != 3 or len(cost) < 2:
        raise ValueError('Expected [streams, windows, candidates], >=2 streams')
    result = []
    for held in range(len(cost)):
        train = np.delete(cost, held, axis=0).mean(0)
        selected = train.argmin(-1)
        optimistic = cost[held].argmin(-1)
        ix = np.arange(cost.shape[1])
        result.append(dict(selected=selected.tolist(),
                           delta=(cost[held, ix, selected]-cost[held, :, 0]).tolist(),
                           same_stream_delta=(cost[held, ix, optimistic]-cost[held, :, 0]).tolist(),
                           agreement=float(np.mean(selected == optimistic))))
    return result


def main():
    source = Path('adaptive_search_results/full_root_distribution.json')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text()); prior = np.asarray(data['policies']['zero'])
    costs = []
    for i, moment in enumerate(data['moments']):
        cost, policies = candidate_costs(np.asarray(moment['attraction']),
                                         np.asarray(moment['pair_distance']), prior)
        np.testing.assert_array_equal(cost[:, 0], data['runs']['zero'][i]['window_cost'])
        costs.append(cost)
    rows = cross_stream(np.asarray(costs)); video = np.asarray(data['video'])
    delta = np.asarray([r['delta'] for r in rows])
    summary = dict(delta=float(delta.mean()), seed_delta=delta.mean(1).tolist(),
                   same_stream_delta=float(np.mean([r['same_stream_delta'] for r in rows])),
                   selection_agreement=float(np.mean([r['agreement'] for r in rows])),
                   unchanged_fraction=float(np.mean(np.asarray([r['selected'] for r in rows]) == 0)),
                   per_video_delta={str(v): float(delta[:, video == v].mean()) for v in np.unique(video)})
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(summary=summary, folds=rows, source_sha256=digest,
                  seed=[m['seed'] for m in data['moments']], video=data['video'], start=data['start'],
                  note='TRAIN-only cached80 roots/451017-20/P4 per component. Fixed9 candidates: prior or .75prior+.25onehot. Select using mean of other3 streams, evaluate omitted stream; future truth SHARED, so oracle only, NOT causal policy/generalization/strict bound. Folds overlap; no independent-seed SE. Same-stream oracle optimistic. No rollout, fit, hold, DEV, TEST, tuning, or promotion; zero score exact replay.')
    Path('adaptive_search_results/root_selection_space.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
