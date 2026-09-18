"""Future-truth oracle diagnostic only; never a deployable gate or validation result."""
import hashlib
import json
from pathlib import Path
import numpy as np
from soft_kernel_gate import scalar_optimum
from soft_output_mixture import value_gradient


def main():
    root=Path('adaptive_search_results');source=root/'soft_kernel_gate_evaluation.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();d=json.loads(source.read_text())
    seeds=sorted({r['seed'] for r in d['runs']});assert len(seeds)==2
    terms={seed:{key:np.zeros((80,3)) for key in ['baseline','linear','quadratic']} for seed in seeds}
    for row in d['runs']:
        for key in terms[row['seed']]:terms[row['seed']][key][row['indices']]=row['coefficients']['4097'][key]
    alphas={}
    for seed in seeds:
        t=terms[seed]
        alphas[seed]=np.array([[scalar_optimum([t['linear'][j,k]],[t['quadratic'][j,k]]) for k in range(3)] for j in range(80)])
    cases=[]
    for fit_seed in seeds:
        for eval_seed in seeds:
            delta=value_gradient(terms[eval_seed],alphas[fit_seed])[0]-terms[eval_seed]['baseline']
            cases.append(dict(oracle_seed=fit_seed,evaluation_seed=eval_seed,
                              same_seed_optimistic=fit_seed==eval_seed,mean_delta=float(delta.mean()),
                              horizon_delta=delta.mean(0).tolist(),zero_alpha=int((alphas[fit_seed]==0).sum()),
                              one_alpha=int((alphas[fit_seed]==1).sum())))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(source_hash=digest,cases=cases,alpha={str(s):a.tolist() for s,a in alphas.items()},
                alpha_cross_rng_correlation=float(np.corrcoef(alphas[seeds[0]].ravel(),alphas[seeds[1]].ravel())[0,1]),
                note='Posthoc FUTURE-TRUTH per-window/per-horizon oracle. Cross-RNG transfer still uses same window futuretruth, '
                     'not deployable/no held-label fitting claim/no new generalization evidence. Same-seed oracle optimistic. '
                     'Independent RNG transfer not rigorous oracle bound. Diagnostic headroom only;no model update/hold/DEV/TEST.')
    (root/'soft_gate_oracle_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(cases=cases,correlation=report['alpha_cross_rng_correlation']),indent=2))


if __name__=='__main__':main()
