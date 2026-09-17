"""Nested TRAIN coverage comparison, fixed critic and reused evaluation labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation, trajectory_cost
from train_search_value import features
from training_window_sampler import TrainingPrefixPool


def candidate_features(s, h):
    q, mem = s.machine.initialize(h)
    pe, tr, read = s.machine.read(h, q, mem)
    xx = []
    for r in range(8):
        y = s.base.execute_rule(h, q, np.full(len(h), r))
        child = np.concatenate([h[:, 1:], y[:, None]], 1)
        xx.append(np.c_[features(s.base, child, np.full(len(h), r), read['read_hidden']), np.eye(8)[q]])
    return np.stack(xx, 1), np.einsum('be,ber->br', pe, tr)


def predict_crossfit(x, target, videos, eval_x, eval_videos, contextual=True):
    scores = np.empty(eval_x.shape[:2])
    for video in np.unique(eval_videos):
        fit = videos != video
        if not fit.any():
            raise ValueError('No independent fitting videos')
        centered = target[fit]-target[fit].mean(1, keepdims=True)
        if not contextual:
            # Equal video weight, even if a future caller has unequal counts.
            score = np.mean([centered[videos[fit]==v].mean(0) for v in np.unique(videos[fit])], 0)
            scores[eval_videos==video] = score
            continue
        raw = x[fit].reshape(-1, x.shape[-1])
        mean, scale = raw.mean(0), np.maximum(raw.std(0), 1e-5)
        projection = np.random.default_rng(1901).normal(size=(x.shape[-1], 128))/np.sqrt(x.shape[-1])
        def design(values):
            z = np.clip((values-mean)/scale, -8, 8)
            phi = np.concatenate([z, np.tanh(z@projection)], -1)
            return phi-phi.mean(1, keepdims=True)
        phi = design(x[fit]); a = phi.reshape(-1, phi.shape[-1])
        coef = np.linalg.solve(a.T@a+len(a)*.01*np.eye(a.shape[1]), a.T@centered.reshape(-1))
        scores[eval_videos==video] = design(eval_x[eval_videos==video])@coef
    return scores


def extra_windows(pool, old, count=24, seed=411017):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ['history', 'truth', 'video', 'start']}
    for video in pool.videos:
        p = pool.pool[video]
        eligible = np.setdiff1d(p['starts'], old['start'][old['video']==video])
        starts = rng.choice(eligible, count, replace=False)
        for t in starts:
            out['history'].append(p['y'][t-31:t+1])
            out['truth'].append(p['y'][t+1:t+301])
            out['video'].append(video); out['start'].append(t)
    return {k: np.asarray(v) for k, v in out.items()}


def summarize(scores, evaluation, prior, video):
    indices = np.arange(len(video)); reference = (evaluation*prior).sum(-1)
    choice = scores.argmin(1); cost = evaluation[:, indices, choice]
    target = evaluation.mean(0); target -= target.mean(1, keepdims=True)
    return dict(cost=float(cost.mean()), delta=float((cost-reference).mean()),
                seed_delta=(cost-reference).mean(1).tolist(),
                per_video_delta={str(v): float((cost-reference)[:, video==v].mean()) for v in np.unique(video)},
                centered_correlation=float(np.corrcoef(scores.ravel(), target.ravel())[0, 1]),
                choice_counts=np.bincount(choice, minlength=8).tolist())


def main():
    root = Path('adaptive_search_results'); source = root/'crossfit_value_audit.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    old = json.loads(source.read_text()); labels = np.array(old['labels'])
    engine = AdaptiveBeam(); pool = TrainingPrefixPool(engine.base, steps=300)
    w = pool.sample(401017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
    x, prior = candidate_features(engine, w['history'])
    np.testing.assert_array_equal(prior, old['prior'])
    small_y = labels[:2].mean(0)
    scores = {'small_context': predict_crossfit(x, small_y, w['video'], x, w['video']),
              'small_action_only': predict_crossfit(x, small_y, w['video'], x, w['video'], False)}
    np.testing.assert_allclose(scores['small_context'], old['scores'], rtol=0, atol=1e-12)
    extra = extra_windows(pool, w)
    new_x, _ = candidate_features(engine, extra['history'])
    new_labels, guards = [], []
    for seed in (412017, 412018):
        costs = []; counts = []
        for r in range(8):
            p, f = continuation(engine, extra['history'], seed, root=r)
            costs.append(trajectory_cost(p, extra['truth'], f)); counts.append(int(f.any(-1).sum()))
        new_labels.append(np.stack(costs, 1)); guards.append(counts)
        print('extra labels', seed, 'mean', np.mean(costs), 'guards', sum(counts), flush=True)
    train_x = np.concatenate([x, new_x]); train_y = np.concatenate([small_y, np.mean(new_labels, 0)])
    train_videos = np.concatenate([w['video'], extra['video']])
    for context in (True, False):
        name = 'dense_context' if context else 'dense_action_only'
        scores[name] = predict_crossfit(train_x, train_y, train_videos, x, w['video'], context)
    summary = {name: summarize(value, labels[2:], prior, w['video']) for name, value in scores.items()}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(summary=summary, scores={k: v.tolist() for k, v in scores.items()},
                  new_labels=np.asarray(new_labels).tolist(), new_video=extra['video'].tolist(), new_start=extra['start'].tolist(),
                  source_sha256=digest, guards=guards, old_score_replay_atol=1e-12,
                  note='Nested8->32windows/video,10fitTRAINprefix videos. Extra24 starts/video seed411017 exclude old exact starts,overlapping intervals allowed. New labels412017/18/P4/300; old8labels retained. Same critic128random features/ridge .01*N. Leave-one-video-out for every train transform and target; action-only means equal video weight. Evaluate old80windows and reused403017/18labels,not new independent confirmation. Coverage increases compute4x,not equal-budget trial. Root-only MSE,not U-energy or repeated controller. No hyperparameter selection/hold/DEV/TEST/default promotion.')
    (root/'value_coverage_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
