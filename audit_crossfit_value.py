"""TRAIN-video-cross-fitted root action values under stochastic continuation."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from feedback_distribution_pilot import sample
from train_search_value import features
from training_window_sampler import TrainingPrefixPool


def trajectory_cost(prediction, truth, failed):
    costs = []
    for t in (50, 100, 300):
        x = prediction[:, :, t-1]
        y = truth[:, None, t-1]
        error = ((np.sin(x)-np.sin(y))**2 + (np.cos(x)-np.cos(y))**2).sum(-1)/4
        costs.append((error + 2*failed[:, :, t-1]).mean(1))
    return np.mean(costs, axis=0)


def continuation(s, histories, seed, particles=4, root=None, forced_step=0):
    """Original sampler; optional one-time forced destination at zero-based step."""
    if not isinstance(forced_step,(int,np.integer)) or not 0<=forced_step<300:
        raise ValueError('forced_step must be an integer in [0,300)')
    h = np.repeat(histories, particles, axis=0)
    q, mem = s.machine.initialize(h)
    rng = np.random.default_rng(seed)
    dead = np.zeros(len(h), bool)
    predictions, failures = [], []
    for t in range(300):
        pe, tr, read = s.machine.read(h, q, mem)
        e = sample(pe, rng.random(len(h)))
        r = sample(tr[np.arange(len(h)), e], rng.random(len(h)))
        if t == forced_step and root is not None:
            r = np.full(len(h), root, dtype=int)
        y = s.base.execute_rule(h, q, r)
        dead |= (~np.isfinite(y)).any(1) | (np.abs(y-h[:, -1]) > np.pi).any(1)
        y[dead] = h[dead, -1]
        predictions.append(y.copy()); failures.append(dead.copy())
        h = np.concatenate([h[:, 1:], y[:, None]], 1)
        q, mem = r, {'hidden': read['read_hidden']}
    return (np.stack(predictions, 1).reshape(len(histories), particles, 300, 2),
            np.stack(failures, 1).reshape(len(histories), particles, 300))


def crossfit(x, target, videos):
    """Fit centered action differences; all preprocessing excludes held video."""
    scores = np.empty_like(target)
    for video in np.unique(videos):
        fit = videos != video
        raw = x[fit].reshape(-1, x.shape[-1])
        mean, scale = raw.mean(0), np.maximum(raw.std(0), 1e-5)
        z = np.clip((x-mean)/scale, -8, 8)
        projection = np.random.default_rng(1901).normal(size=(x.shape[-1], 128))/np.sqrt(x.shape[-1])
        phi = np.concatenate([z, np.tanh(z@projection)], -1)
        phi -= phi.mean(1, keepdims=True)
        centered = target-target.mean(1, keepdims=True)
        a = phi[fit].reshape(-1, phi.shape[-1])
        b = centered[fit].reshape(-1)
        coef = np.linalg.solve(a.T@a + len(a)*.01*np.eye(a.shape[1]), a.T@b)
        scores[~fit] = phi[~fit]@coef
    return scores


def main():
    s = AdaptiveBeam(); root = Path('adaptive_search_results')
    w = TrainingPrefixPool(s.base, steps=300).sample(401017, per_video=8)
    h = w['history']; q, mem = s.machine.initialize(h)
    pe, tr, read = s.machine.read(h, q, mem)
    prior = np.einsum('be,ber->br', pe, tr)
    xx = []
    for r in range(8):
        y = s.base.execute_rule(h, q, np.full(len(h), r))
        ch = np.concatenate([h[:, 1:], y[:, None]], 1)
        xx.append(np.c_[features(s.base, ch, np.full(len(h), r), read['read_hidden']), np.eye(8)[q]])
    x = np.stack(xx, 1)
    # Exact unchecked baseline parity before forced-action interventions.
    from continuous_residual_pilot import rollout
    small = {k: v[:1] for k, v in w.items()}
    _, old, oldf = rollout(s, small, None, 402017, particles=4)
    p, f = continuation(s, small['history'], 402017)
    np.testing.assert_array_equal(p, old); np.testing.assert_array_equal(f, oldf)
    labels = []; guards = []
    for seed in (402017, 402018, 403017, 403018):
        costs = []; counts = []
        for r in range(8):
            p, f = continuation(s, h, seed, root=r)
            costs.append(trajectory_cost(p, w['truth'], f)); counts.append(int(f.any(-1).sum()))
        labels.append(np.stack(costs, 1)); guards.append(counts)
        print('label seed', seed, 'mean', np.mean(costs), 'guards', sum(counts), flush=True)
    labels = np.array(labels)
    fit_target = labels[:2].mean(0); evaluation = labels[2:].mean(0)
    scores = crossfit(x, fit_target, w['video'])
    choices = {'crossfit_value': scores.argmin(1), 'prior_mode': prior.argmax(1),
               'fit_label_oracle': fit_target.argmin(1), 'evaluation_oracle': evaluation.argmin(1)}
    idx = np.arange(len(h)); reference = (prior*evaluation).sum(1)
    summary = {}
    for name, choice in choices.items():
        cost = evaluation[idx, choice]
        summary[name] = dict(cost=float(cost.mean()), delta_vs_frozen_sampling=float((cost-reference).mean()),
                             per_video_delta={str(v): float((cost-reference)[w['video']==v].mean()) for v in np.unique(w['video'])},
                             evaluation_seed_costs=[float(y[idx, choice].mean()) for y in labels[2:]])
    centered_fit = fit_target-fit_target.mean(1, keepdims=True)
    centered_eval = evaluation-evaluation.mean(1, keepdims=True)
    report = dict(summary=summary, frozen_sampling_cost=float(reference.mean()),
                  independent_label_advantage_correlation=float(np.corrcoef(centered_fit.ravel(), centered_eval.ravel())[0, 1]),
                  prediction_advantage_correlation=float(np.corrcoef(scores.ravel(), centered_eval.ravel())[0, 1]),
                  scores=scores.tolist(), labels=labels.tolist(), prior=prior.tolist(),
                  video=w['video'].tolist(), start=w['start'].tolist(), guards=guards, baseline_exact_parity=True,
                  note='80 TRAIN-fit-prefix windows,8/video; all8 forced root destinations,normal stochastic continuation/P4/300. Root e has no separate future effect given r in frozen engine. MSE sincos4D+2failure averaged50/100/300,NOT U-energy. Two fit RNG402017/18,two independent evaluation RNG403017/18,common RNG across candidates. Leave-one-video-out centered ridge(alpha .01*N)+128 fixed random features,preprocessing excludes video,no tuning. Frozen sampling reference=sum_r prior[r]*conditional cost,not separate sampled rollout. Fit-label oracle uses same-window truth and is undeployable but evaluated on independent RNG; evaluation oracle optimistically selected on evaluation labels. Observed root histories only,one root intervention,not repeated closed-loop controller. Backbone trained on these videos; only critic fit is video-separated. No hold/DEV/TEST/default promotion.')
    (root/'crossfit_value_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['summary', 'frozen_sampling_cost', 'independent_label_advantage_correlation', 'prediction_advantage_correlation']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
