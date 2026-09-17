"""Matched absolute-position versus transported-increment supervision."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import collect, fit_models, observed_score, rollout
from drift_residual_pilot import generated_rows
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def transport_targets(observed, generated):
    """Offline labels only: preserve observed angular increment at generated angle."""
    if not np.array_equal(observed['video'],generated['video']):
        raise ValueError('Observed/generated rows must be aligned')
    increment=np.angle(np.exp(1j*(observed['truth']-observed['history'][:,-1])))
    result={k:v.copy() for k,v in generated.items()}
    result['truth']=generated['history'][:,-1]+increment
    return result


def main():
    root=Path('adaptive_search_results');s=AdaptiveBeam()
    observed=collect(s);generated=generated_rows(s,observed);transported=transport_targets(observed,generated)
    original,oc,of=fit_models(s.base,observed)
    models={'zero':None,'observed':original['ridge_0.0001']}
    for label,data in [('absolute',generated),('increment',transported)]:
        mixed={k:np.concatenate([observed[k],data[k]]) for k in observed}
        fitted,_,_=fit_models(s.base,mixed)
        model=fitted['ridge_0.0001'];model['cap']=models['observed']['cap'].copy()
        models[label]=model
        np.savez(root/f'increment_residual_model_{label}.npz',**model)
    for key in ['mean','scale','cap']:
        np.testing.assert_array_equal(models['absolute'][key],models['increment'][key])
    for current,previous in [('observed','observed'),('absolute','mixed')]:
        with np.load(root/f'drift_residual_model_{previous}.npz') as saved:
            for key,value in models[current].items():np.testing.assert_array_equal(value,saved[key])
    _,gc,gf=fit_models(s.base,generated)
    diagnostics={name:{label:{part:observed_score(s,data,centers,model,mask)
                             for part,mask in [('fit',fit),('hold',~fit)]}
                      for label,data,centers,fit in [('observed',observed,oc,of),
                          ('generated_absolute',generated,gc,gf),('generated_increment',transported,gc,gf)]}
                 for name,model in models.items()}
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    free={name:[] for name in models}
    for seed in [32017,32018,32019,32020]:
        for name,model in models.items():
            result,pred,failed=rollout(s,hold,model,seed);free[name].append(result)
            if name!='increment':
                oldname={'absolute':'mixed'}.get(name,name)
                with np.load(root/f'drift_residual_hold_{oldname}_{seed}.npz') as old:
                    for key,value in [('prediction',pred),('failed',failed),('truth',hold['truth']),
                                      ('video',hold['video']),('window_start',hold['start'])]:
                        np.testing.assert_array_equal(value,old[key])
            np.savez_compressed(root/f'increment_residual_hold_{name}_{seed}.npz',prediction=pred,
                                failed=failed,truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,result['objective'],flush=True)
    report=dict(diagnostics=diagnostics,holdout=free,alpha=.0001,
                common_cap=models['observed']['cap'].tolist(),normalization_exactly_equal=True,
                previous_models_exact_replay=True,previous_rollouts_exact_replay=True,
                mean_objective={k:float(np.mean([r['objective'] for r in v])) for k,v in free.items()},
                note='Only generated-state target changes between absolute/increment. Same states, weights, normalization, cap, alpha and RNG. Transported increments are offline hypothetical labels, not physical oracle at displaced state. TRAIN10fit/3reusedhold only, no DEV/TEST. Same hold seeds deliberately paired with prior experiment, not fresh validation. No checker enforced, no default promotion.')
    (root/'increment_residual_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
