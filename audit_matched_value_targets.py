"""Matched causal critics differing only in target; independent U-energy checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import action_labels, centered_correlation
from audit_value_coverage import candidate_features, predict_crossfit
from audit_soft_value import calibrate, tilt, conditional_kl, read_probabilities
from training_window_sampler import TrainingPrefixPool


def fit_policy(x, target, videos, pe, tr):
    scores = np.empty_like(target); probability = np.empty_like(target)
    kl = np.empty(len(x)); folds = {}
    for video in np.unique(videos):
        fit = videos != video; held = ~fit
        xx = np.concatenate([x[fit], x[held]])
        prediction = predict_crossfit(x, target, videos, xx, np.full(len(xx), video))
        fs, hs = prediction[:fit.sum()], prediction[fit.sum():]
        strength, train_kl, capped = calibrate(pe[fit], tr[fit], fs)
        changed = tilt(tr[held], hs, strength)
        probability[held] = np.einsum('be,ber->br', pe[held], changed)
        scores[held] = hs; kl[held] = conditional_kl(pe[held], tr[held], changed)
        folds[str(video)] = dict(strength=strength, train_mean_kl=train_kl, capped=capped)
    return dict(scores=scores, probability=probability, kl=kl, folds=folds)


def evaluate(models, labels, prior, videos):
    baseline = (labels*prior).sum(-1); summary = {}
    for name, model in models.items():
        scores = model['scores']; choice = scores.argmin(1)
        for kind, costs in [('hard', labels[:, np.arange(len(prior)), choice]),
                            ('soft', (labels*model['probability']).sum(-1))]:
            delta = costs-baseline
            summary[name+'_'+kind] = dict(cost=float(costs.mean()), delta=float(delta.mean()),
                                          seed_delta=delta.mean(1).tolist(),
                                          per_video_delta={str(v): float(delta[:, videos==v].mean()) for v in np.unique(videos)})
        summary[name+'_correlation'] = centered_correlation(scores, labels.mean(0))
    return dict(baseline=float(baseline.mean()), models=summary)


def main():
    root = Path('adaptive_search_results'); source = root/'energy_action_value_audit.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); old = json.loads(source.read_text())
    engine = AdaptiveBeam(); w = TrainingPrefixPool(engine.base, steps=300).sample(401017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(w[key], old[key])
    x, prior = candidate_features(engine, w['history']); pe, tr = read_probabilities(engine, w['history'])
    np.testing.assert_array_equal(prior, old['prior'])
    targets = {name: np.array([r[key] for r in old['records'][:2]]).mean(0)
               for name, key in [('energy', 'costs'), ('mse', 'mse'), ('attraction', 'attraction')]}
    models = {name: fit_policy(x, target, w['video'], pe, tr) for name, target in targets.items()}
    # All targets/policies and strengths fixed before either evaluation phase.
    reused = np.array([r['costs'] for r in old['records'][2:]])
    old_result = evaluate(models, reused, prior, w['video'])
    print('reused diagnostic', json.dumps(old_result), flush=True)
    records = []
    for seed in range(441017, 441021):
        base, bf = continuation(engine, w['history'], seed)
        values, guards = [], []
        for r in range(8):
            candidate, cf = continuation(engine, w['history'], seed, root=r)
            cost, _, _ = action_labels(base, bf, candidate, cf, w['truth'])
            values.append(cost); guards.append(int(cf.any(-1).sum()))
        records.append(dict(seed=seed, costs=np.stack(values, 1).tolist(), guards=guards,
                            baseline_guards=int(bf.any(-1).sum())))
        print('new U labels', seed, 'guards', sum(guards), flush=True)
    fresh = evaluate(models, np.array([r['costs'] for r in records]), prior, w['video'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    serial = {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}
              for name, model in models.items()}
    report = dict(reused_evaluation=old_result, fresh_evaluation=fresh, models=serial, records=records,
                  source_sha256=digest, video=w['video'].tolist(), start=w['start'].tolist(),
                  note='Matched10fitTRAINvideo LOVO/80roots/samefeatures128random/ridge .01*N. Only label target differs:full single-particle U,trajectory MSE,attraction. Fit431017/18; old431019/20 reused diagnostic then ALL frozen candidates new441017-20. Hardargmin and event-conditional soft tilt(.01meanfoldtrainKL) retained,no selection or temperature tuning. Source U label scaling is1/P;centered ridge linear,soft calibration compensates positive scale absent cap. Only one particle root changes,others original,not collective or repeated policy improvement. No hold/DEV/TEST/default promotion.')
    (root/'matched_value_targets.json').write_text(json.dumps(report, indent=2))
    print('fresh', json.dumps(fresh, indent=2), flush=True)


if __name__ == '__main__':
    main()
