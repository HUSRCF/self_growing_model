"""Matched causal critics for first-order vs full fixed-mixture improvement."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_value_coverage import candidate_features, predict_crossfit
from audit_root_selection_space import candidate_costs
from audit_full_root_distribution import change_terms, mixture_score
from confirm_root_mixture_holdout import collect_moments
from training_window_sampler import TrainingPrefixPool


def targets(a, b, prior):
    cost, policies = candidate_costs(a, b, prior)
    full = cost-cost[:, :1]
    linear = np.stack([change_terms(a, b, policies[:, i], prior)[0]
                       for i in range(policies.shape[1])], 1)
    return linear, full, policies


def policy_features(x, prior):
    # Same feature map for both objectives; prior represented by weighted child features.
    center = np.einsum('nr,nrd->nd', prior, x)
    return np.concatenate([center[:, None], .75*center[:, None]+.25*x], 1)


def worker(args):
    w, seed = args
    return seed, collect_moments(AdaptiveBeam(), w, seed, 8)


def main():
    root = Path('adaptive_search_results'); source = root/'full_root_distribution.json'
    files = [source, Path('v20_rnn_mixture/models/gru_1901.npz'),
             Path('v20_rnn_mixture/models/frozen_dynamics.json')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    old = json.loads(source.read_text()); engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(401017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
    x, prior = candidate_features(engine, w['history'])
    np.testing.assert_array_equal(prior, old['policies']['zero'])
    rows = []
    for i, moment in enumerate(old['moments']):
        a, b = np.asarray(moment['attraction']), np.asarray(moment['pair_distance'])
        np.testing.assert_array_equal(mixture_score(a, b, prior), old['runs']['zero'][i]['window_cost'])
        linear, full, policies = targets(a, b, prior); rows.append([linear, full])
    labels = np.mean(rows, axis=0); xx = policy_features(x, prior)
    models = {}
    for j, target in enumerate(['linear', 'full']):
        for contextual in [False, True]:
            name = target + ('_context' if contextual else '_action')
            scores = predict_crossfit(xx, labels[j], w['video'], xx, w['video'], contextual)
            choice = scores.argmin(1)
            models[name] = dict(scores=scores.tolist(), choice=choice.tolist(),
                                probability=policies[np.arange(len(x)), choice].tolist())
    frozen = dict(models=models, source_hashes=hashes, video=w['video'].tolist(), start=w['start'].tolist(),
                  note='10-video LOVO, same9 candidates/features/random128/ridge .01N; only target linear vs full U changes. Both retain prior. Labels mean451017-20/P4; no tuning. All frozen before new691017/18 P8 evaluation. Causal inference; backbone already TRAIN, adapter-level CV only.')
    model_path = root/'root_mixture_critic_model.json'
    model_path.write_text(json.dumps(frozen, indent=2))
    replay = json.loads(model_path.read_text())
    for name in models:
        np.testing.assert_array_equal(replay['models'][name]['probability'], models[name]['probability'])
    print('All four policies frozen before evaluation', flush=True)
    runs = []; summaries = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for seed, (a, b, terms, guards) in pool.map(worker, [(w, s) for s in [691017, 691018]]):
            baseline = mixture_score(a, b, prior); results = {}
            for name, model in models.items():
                probability = np.asarray(model['probability'])
                delta = mixture_score(a, b, probability)-baseline
                results[name] = dict(delta=delta.tolist(), mean=float(delta.mean()),
                                     horizon_delta=[float((mixture_score(ta,tb,probability)-mixture_score(ta,tb,prior)).mean()) for ta,tb in terms])
            runs.append(dict(seed=seed, baseline=baseline.tolist(), results=results, guards=guards,
                             attraction=a.tolist(), pair_distance=b.tolist()))
            print(seed, {k: v['mean'] for k,v in results.items()}, 'guards', sum(guards), flush=True)
    for name in models:
        d = np.array([r['results'][name]['delta'] for r in runs])
        summaries[name] = dict(delta=float(d.mean()), seed_delta=d.mean(1).tolist(),
                               per_video_delta={str(v): float(d[:,w['video']==v].mean()) for v in np.unique(w['video'])},
                               horizon_delta=np.mean([r['results'][name]['horizon_delta'] for r in runs],0).tolist(),
                               unchanged_fraction=float(np.mean(np.asarray(models[name]['choice'])==0)))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = dict(summary=summaries, baseline=float(np.mean([r['baseline'] for r in runs])), runs=runs,
                  source_hashes=hashes, note=frozen['note']+' No hold/DEV/TEST, no repeated-step control or promotion; paired new RNG not new video dataset.')
    (root/'root_mixture_critic_evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
