"""Single-particle root action values for the actual ensemble U-energy."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation, trajectory_cost
from audit_value_coverage import candidate_features
from ensemble_score_objective import energy_costs
from feedback_distribution_pilot import sample
from training_window_sampler import TrainingPrefixPool


def replacement_delta(x, y, failed, candidate, candidate_failed, index):
    """Exact change when only particle index is replaced, no LOO renormalization."""
    p = x.shape[1]
    if p < 3 or not 0 <= index < p:
        raise ValueError('At least three particles and valid index required')
    others = np.arange(p) != index
    attraction = np.linalg.norm(candidate-y, axis=-1)-np.linalg.norm(x[:, index]-y, axis=-1)
    spread = (np.linalg.norm(candidate[:, None]-x[:, others], axis=-1)
              -np.linalg.norm(x[:, index, None]-x[:, others], axis=-1)).sum(1)/(p-1)
    failure = 2*(candidate_failed.astype(float)-failed[:, index].astype(float))
    return (attraction-spread+failure)/p


def embedding(x):
    return np.concatenate([np.sin(x), np.cos(x)], -1)


def action_labels(base, base_failed, candidate, candidate_failed, truth):
    """Average four SEPARATE one-particle interventions, not all-particle forcing."""
    values, attractions, spreads = [], [], []
    for t in (50, 100, 300):
        x, c, y = embedding(base[:, :, t-1]), embedding(candidate[:, :, t-1]), embedding(truth[:, t-1])
        f, cf = base_failed[:, :, t-1], candidate_failed[:, :, t-1]
        original, _ = energy_costs(x, y, f)
        replacements, spread_terms = [], []
        for i in range(x.shape[1]):
            delta = replacement_delta(x, y, f, c[:, i], cf[:, i], i)
            replaced = x.copy(); replaced[:, i] = c[:, i]
            replaced_failed = f.copy(); replaced_failed[:, i] = cf[:, i]
            direct, _ = energy_costs(replaced, y, replaced_failed)
            np.testing.assert_allclose(original+delta, direct, rtol=0, atol=1e-14)
            replacements.append(direct)
            spread_terms.append(np.linalg.norm(c[:, i, None]-x[:, np.arange(x.shape[1])!=i], axis=-1).mean(1))
        values.append(np.mean(replacements, 0))
        attractions.append((np.linalg.norm(c-y[:, None], axis=-1)+2*cf).mean(1))
        spreads.append(np.mean(spread_terms, 0))
    return np.mean(values, 0), np.mean(attractions, 0), np.mean(spreads, 0)


def centered_correlation(a, b):
    a = a-a.mean(-1, keepdims=True); b = b-b.mean(-1, keepdims=True)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    root = Path('adaptive_search_results'); engine = AdaptiveBeam()
    w = TrainingPrefixPool(engine.base, steps=300).sample(401017, per_video=8)
    _, prior = candidate_features(engine, w['history'])
    q, mem = engine.machine.initialize(np.repeat(w['history'], 4, axis=0))
    pe, tr, _ = engine.machine.read(np.repeat(w['history'], 4, axis=0), q, mem)
    records = []; matching_paths = 0
    for seed in range(431017, 431021):
        base, bf = continuation(engine, w['history'], seed)
        rng = np.random.default_rng(seed)
        e = sample(pe, rng.random(len(q))); r0 = sample(tr[np.arange(len(q)), e], rng.random(len(q))).reshape(-1, 4)
        values, attractions, spreads, mse, guards = [], [], [], [], []
        for r in range(8):
            candidate, cf = continuation(engine, w['history'], seed, root=r)
            match = r0==r
            np.testing.assert_array_equal(candidate[match], base[match])
            np.testing.assert_array_equal(cf[match], bf[match])
            matching_paths += int(match.sum())
            cost, attraction, spread = action_labels(base, bf, candidate, cf, w['truth'])
            values.append(cost); attractions.append(attraction); spreads.append(spread)
            mse.append(trajectory_cost(candidate, w['truth'], cf)); guards.append(int(cf.any(-1).sum()))
        values, attractions, spreads, mse = [np.stack(a, 1) for a in (values, attractions, spreads, mse)]
        # Candidate-centered scaled full cost must equal attraction minus spread.
        np.testing.assert_allclose(4*(values-values.mean(1, keepdims=True)),
                                   (attractions-spreads)-(attractions-spreads).mean(1, keepdims=True), rtol=0, atol=1e-14)
        records.append(dict(seed=seed, costs=values.tolist(), attraction=attractions.tolist(), spread=spreads.tolist(),
                            mse=mse.tolist(), guard_particles=guards, baseline_guard_particles=int(bf.any(-1).sum())))
        print(seed, 'full U mean', values.mean(), 'guards', sum(guards), flush=True)
    values = np.array([r['costs'] for r in records]); mse = np.array([r['mse'] for r in records])
    attraction = np.array([r['attraction'] for r in records]); spread = np.array([r['spread'] for r in records])
    fit, evaluation = values[:2].mean(0), values[2:].mean(0)
    choices = {'fit_u_oracle': fit.argmin(1), 'fit_mse_oracle': mse[:2].mean(0).argmin(1),
               'fit_attraction_oracle': attraction[:2].mean(0).argmin(1),
               'prior_mode': prior.argmax(1), 'evaluation_u_oracle': evaluation.argmin(1)}
    reference = (values[2:]*prior).sum(-1); summary = {}
    for name, choice in choices.items():
        cost = values[2:, np.arange(len(prior)), choice]; delta = cost-reference
        summary[name] = dict(cost=float(cost.mean()), delta=float(delta.mean()), seed_delta=delta.mean(1).tolist(),
                             per_video_delta={str(v): float(delta[:, w['video']==v].mean()) for v in np.unique(w['video'])})
    report = dict(records=records, summary=summary, prior=prior.tolist(), baseline_cost=float(reference.mean()),
                  repeat_correlation=dict(energy=centered_correlation(fit, evaluation),
                                          mse=centered_correlation(mse[:2].mean(0), mse[2:].mean(0)),
                                          attraction=centered_correlation(attraction[:2].mean(0), attraction[2:].mean(0)),
                                          spread=centered_correlation(spread[:2].mean(0), spread[2:].mean(0))),
                  cross_objective_correlation=centered_correlation(values.mean(0), mse.mean(0)),
                  fit_u_vs_mse_choice_disagreements=int((choices['fit_u_oracle']!=choices['fit_mse_oracle']).sum()),
                  matching_baseline_paths=matching_paths, video=w['video'].tolist(), start=w['start'].tolist(),
                  note='Same80fitTRAINroot windows,4newRNG431017-20/P4/300. Each candidate changes ONE particle root,other3 original independent sampled trajectories; average four separate interventions. Common continuation uniforms,matching root gives exact original trajectory. Full energy direct replay+replacement algebra checked each horizon/window/particle. First2RNG labels select oracle,second2evaluate,not train a critic;oracles use futuretruth and are undeployable. Root prior weighted baseline is conditional-root integration estimate,not bitwise sampled-ensemble score. Single-particle intervention NOT simultaneous all-particle policy improvement. No hold/DEV/TEST/tuning/promotion.')
    (root/'energy_action_value_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['summary', 'baseline_cost', 'repeat_correlation', 'cross_objective_correlation', 'fit_u_vs_mse_choice_disagreements', 'matching_baseline_paths']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
