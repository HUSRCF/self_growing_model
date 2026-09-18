"""Refit all three fixed critics on ten videos; evaluate three TRAIN holdouts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_value_coverage import candidate_features
from audit_soft_value import calibrate, tilt, conditional_kl, read_probabilities
from audit_full_root_distribution import mixture_terms, mixture_score, change_terms
from training_window_sampler import TrainingPrefixPool
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def design(model, x):
    z = np.clip((x-model['mean'])/model['scale'], -8, 8)
    phi = np.concatenate([z, np.tanh(z@model['projection'])], -1)
    return phi-phi.mean(1, keepdims=True)


def fit_critic(x, target):
    raw = x.reshape(-1, x.shape[-1])
    model = dict(mean=raw.mean(0), scale=np.maximum(raw.std(0), 1e-5),
                 projection=np.random.default_rng(1901).normal(size=(x.shape[-1], 128))/np.sqrt(x.shape[-1]))
    phi = design(model, x); a = phi.reshape(-1, phi.shape[-1])
    y = (target-target.mean(1, keepdims=True)).reshape(-1)
    model['coef'] = np.linalg.solve(a.T@a+len(a)*.01*np.eye(a.shape[1]), a.T@y)
    return model


def predict(model, x):
    return design(model, x)@model['coef']


def main():
    root = Path('adaptive_search_results'); source = root/'energy_action_value_audit.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); old = json.loads(source.read_text())
    engine = AdaptiveBeam(); train = TrainingPrefixPool(engine.base, steps=300).sample(401017, per_video=8)
    for key in ['video', 'start']:
        np.testing.assert_array_equal(train[key], old[key])
    x, _ = candidate_features(engine, train['history']); pe, tr = read_probabilities(engine, train['history'])
    models, calibration = {}, {}
    for name, key in [('energy', 'costs'), ('mse', 'mse'), ('attraction', 'attraction')]:
        target = np.array([r[key] for r in old['records'][:2]]).mean(0)
        model = fit_critic(x, target); score = predict(model, x)
        strength, kl, capped = calibrate(pe, tr, score)
        models[name] = model; calibration[name] = dict(strength=strength, train_mean_kl=kl, capped=capped)
        serialized = json.loads(json.dumps({k: v.tolist() for k, v in model.items()}))
        np.testing.assert_array_equal(predict({k: np.array(v) for k, v in serialized.items()}, x), score)
    # The three critics and strengths are fixed before reading holdout windows.
    w = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
    assert not set(w['video']) & set(train['video'])
    hx, prior = candidate_features(engine, w['history']); hp, ht = read_probabilities(engine, w['history'])
    policies = {'zero': prior}; held_kl = {}; scores = {}
    for name, model in models.items():
        score = predict(model, hx); scores[name] = score.tolist()
        hard = np.eye(8)[score.argmin(1)]
        changed = tilt(ht, score, calibration[name]['strength'])
        policies[name+'_hard'] = hard; policies[name+'_mix25'] = .75*prior+.25*hard
        policies[name+'_soft'] = np.einsum('be,ber->br', hp, changed)
        kl = conditional_kl(hp, ht, changed); held_kl[name] = dict(mean=float(kl.mean()), max=float(kl.max()))
    runs = {name: [] for name in policies}; moments = []
    for seed in range(461017, 461021):
        endpoints, failures, guards = [], [], []
        for r in range(8):
            p, f = continuation(engine, w['history'], seed, root=r)
            endpoints.append(p[:, :, [49, 99, 299]]); failures.append(f[:, :, [49, 99, 299]])
            guards.append(int(f.any(-1).sum()))
        xx, dead = embedding(np.stack(endpoints, 1)), np.stack(failures, 1)
        terms = [mixture_terms(xx[:, :, :, i], embedding(w['truth'][:, t-1]), dead[:, :, :, i])
                 for i, t in enumerate((50, 100, 300))]
        a, b = np.mean([v[0] for v in terms], 0), np.mean([v[1] for v in terms], 0)
        baseline = mixture_score(a, b, prior)
        for name, probability in policies.items():
            cost = mixture_score(a, b, probability); linear, quadratic = change_terms(a, b, probability, prior)
            np.testing.assert_allclose(cost-baseline, linear+quadratic, rtol=0, atol=1e-14)
            horizon_costs = [mixture_score(ta, tb, probability) for ta, tb in terms]
            runs[name].append(dict(seed=seed, cost=float(cost.mean()), delta=float((cost-baseline).mean()),
                                   window_cost=cost.tolist(), horizon_costs=[float(c.mean()) for c in horizon_costs],
                                   linear=float(linear.mean()), quadratic=float(quadratic.mean()),
                                   per_video_delta={str(v): float((cost-baseline)[w['video']==v].mean()) for v in np.unique(w['video'])}))
        moments.append(dict(seed=seed, attraction=a.tolist(), pair_distance=b.tolist(), guards=guards))
        print('TRAIN holdout', seed, {k: v[-1]['delta'] for k, v in runs.items()}, flush=True)
    summary = {}
    for name, rows in runs.items():
        delta = np.array([r['delta'] for r in rows])
        summary[name] = dict(cost=float(np.mean([r['cost'] for r in rows])), delta=float(delta.mean()),
                             conditional_seed_se=float(delta.std(ddof=1)/2), better_seeds=int((delta < 0).sum()),
                             per_video_delta={v: float(np.mean([r['per_video_delta'][v] for r in rows])) for v in rows[0]['per_video_delta']},
                             horizon_costs=np.mean([r['horizon_costs'] for r in rows], 0).tolist())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(summary=summary, runs=runs, moments=moments, policies={k: v.tolist() for k, v in policies.items()},
                  models={name: {k: v.tolist() for k, v in m.items()} for name, m in models.items()},
                  calibration=calibration, held_kl=held_kl, scores=scores, source_sha256=digest,
                  video=w['video'].tolist(), start=w['start'].tolist(),
                  note='All3critics refit on all10fitTRAINprefix/80windows,only431017/18 labels,fixedarchitecture128/ridge .01*N. Serialized prediction exact. MeanKL .01 calibrated fitstates only. Frozen before TRAINhold videos18/16/13 prefix8each,4new461017-20/P4percomponent/300. Same IID root integration and off-diagonal crosscomponent pairs,all10policies retained,fixed25%mix. Hold videos reused historically by other experiments/backbone,not globally blind. No reselection/tuning/DEV/TEST/repeatedcontroller/default promotion.')
    (root/'root_mixture_holdout.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
