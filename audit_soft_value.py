"""Fold-calibrated small-KL event-conditional value tilts, fresh root labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation, trajectory_cost
from audit_value_coverage import candidate_features, extra_windows, predict_crossfit
from training_window_sampler import TrainingPrefixPool


def tilt(transitions, scores, strength):
    t = np.asarray(transitions, float); scores = np.asarray(scores, float)
    if not np.isfinite(strength) or strength < 0 or not np.isfinite(scores).all():
        raise ValueError('Finite scores and nonnegative strength required')
    if t.ndim != 3 or scores.shape != (t.shape[0], t.shape[2]):
        raise ValueError('Expected transitions[N,E,R] and scores[N,R]')
    if (t < 0).any() or not np.isfinite(t).all() or not np.allclose(t.sum(-1), 1):
        raise ValueError('Invalid transition probabilities')
    if strength == 0:
        return t.copy()
    logt = np.full_like(t, -np.inf)
    np.log(t, out=logt, where=t > 0)
    logits = logt-strength*(scores-scores.min(-1, keepdims=True))[:, None, :]
    logits -= logits.max(-1, keepdims=True)
    weights = np.exp(logits)
    return weights/weights.sum(-1, keepdims=True)


def conditional_kl(pe, original, changed):
    terms = np.zeros_like(changed)
    mask = changed > 0
    terms[mask] = changed[mask]*(np.log(changed[mask])-np.log(original[mask]))
    return (pe*terms.sum(-1)).sum(-1)


def calibrate(pe, transitions, scores, budget=.01):
    if budget <= 0:
        raise ValueError('Positive calibration budget required')
    def divergence(strength):
        return float(conditional_kl(pe, transitions, tilt(transitions, scores, strength)).mean())
    low, high = 0., 1.
    while divergence(high) < budget and high < 1e6:
        high = min(high*2, 1e6)
    if divergence(high) < budget:
        return high, divergence(high), True
    for _ in range(60):
        mid = (low+high)/2
        if divergence(mid) <= budget:
            low = mid
        else:
            high = mid
    return low, divergence(low), False


def read_probabilities(engine, histories):
    q, mem = engine.machine.initialize(histories)
    pe, tr, _ = engine.machine.read(histories, q, mem)
    return pe, tr


def main():
    root = Path('adaptive_search_results')
    sources = [root/'crossfit_value_audit.json', root/'value_coverage_audit.json']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    old, coverage = [json.loads(p.read_text()) for p in sources]
    engine = AdaptiveBeam(); pool = TrainingPrefixPool(engine.base, steps=300)
    w = pool.sample(401017, per_video=8); extra = extra_windows(pool, w)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
        np.testing.assert_array_equal(extra[key], coverage['new_'+key])
    x, prior = candidate_features(engine, w['history']); new_x, _ = candidate_features(engine, extra['history'])
    pe, tr = read_probabilities(engine, w['history'])
    np.testing.assert_array_equal(np.einsum('be,ber->br', pe, tilt(tr, np.zeros_like(prior), 0)), old['prior'])
    new_pe, new_tr = read_probabilities(engine, extra['history'])
    small_y = np.array(old['labels'])[:2].mean(0)
    all_x = np.concatenate([x, new_x]); all_y = np.concatenate([small_y, np.array(coverage['new_labels']).mean(0)])
    all_v = np.concatenate([w['video'], extra['video']])
    all_pe, all_tr = np.concatenate([pe, new_pe]), np.concatenate([tr, new_tr])
    policies = {'zero': prior}; calibrations = {}; scores_all = {}; evaluation_kl = {}
    for dense in (False, True):
        tx, ty, tv, tp, tt = (all_x, all_y, all_v, all_pe, all_tr) if dense else (x, small_y, w['video'], pe, tr)
        for contextual in (True, False):
            name = ('dense' if dense else 'small')+('_context' if contextual else '_action_only')
            probability = np.empty_like(prior); scores = np.empty_like(prior); kl = np.empty(len(x)); folds = {}
            for video in np.unique(w['video']):
                held = w['video']==video; fit = tv!=video
                # One fold model supplies training predictions for calibration
                # and held-video predictions; held labels never enter either.
                inputs = np.concatenate([tx[fit], x[held]])
                prediction = predict_crossfit(tx, ty, tv, inputs, np.full(len(inputs), video), contextual)
                fit_score, test_score = prediction[:fit.sum()], prediction[fit.sum():]
                strength, train_kl, capped = calibrate(tp[fit], tt[fit], fit_score)
                changed = tilt(tr[held], test_score, strength)
                probability[held] = np.einsum('be,ber->br', pe[held], changed)
                scores[held] = test_score; kl[held] = conditional_kl(pe[held], tr[held], changed)
                folds[str(video)] = dict(strength=strength, train_mean_kl=train_kl, capped=capped,
                                         fitting_windows=int(fit.sum()))
            np.testing.assert_allclose(scores, coverage['scores'][name], rtol=0, atol=1e-12)
            policies[name] = probability; scores_all[name] = scores; calibrations[name] = folds
            evaluation_kl[name] = dict(mean=float(kl.mean()), max=float(kl.max()), per_window=kl.tolist())
    # Freeze every policy before generating fresh evaluation continuations.
    labels, guards = [], []
    for seed in range(421017, 421021):
        costs, counts = [], []
        for r in range(8):
            p, f = continuation(engine, w['history'], seed, root=r)
            costs.append(trajectory_cost(p, w['truth'], f)); counts.append(int(f.any(-1).sum()))
        labels.append(np.stack(costs, 1)); guards.append(counts)
        print('fresh label', seed, 'guards', sum(counts), flush=True)
    labels = np.array(labels); base = (labels*prior).sum(-1)
    summary = {}; hard = {}
    def summarize(cost):
        delta = cost-base
        return dict(cost=float(cost.mean()), delta=float(delta.mean()), seed_delta=delta.mean(1).tolist(),
                    per_video_delta={str(v): float(delta[:, w['video']==v].mean()) for v in np.unique(w['video'])})
    for name, probability in policies.items():
        summary[name] = summarize((labels*probability).sum(-1))
    for name, scores in scores_all.items():
        hard[name] = summarize(labels[:, np.arange(len(x)), scores.argmin(1)])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p, v in hashes.items())
    report = dict(summary=summary, hard_argmin=hard, calibrations=calibrations, evaluation_kl=evaluation_kl,
                  policies={k: v.tolist() for k, v in policies.items()}, labels=labels.tolist(), guards=guards,
                  source_hashes=hashes, video=w['video'].tolist(), start=w['start'].tolist(),
                  note='Frozen four existing critics,LOVO. Tnew(r|e) proportional Told(r|e)*exp(-lambda*score(r));pe unchanged. Mean joint KL(new||old)=.01 calibrated on each fold TRAIN states only,not perstate/held-video guarantee;lambda cap1e6. Root marginal analytically summed,normal random continuation afterwards. New4RNG421017-20/P4/300 on same80windows,MSE not U-energy. Zero and all hardargmin comparators retained. No temperature tuning/reselection/repeated-policy rollout/hold/DEV/TEST/promotion. Source scores replayatol1e-12.')
    (root/'soft_value_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'soft': summary, 'hard': hard, 'kl': {k: v['mean'] for k, v in evaluation_kl.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
