"""Fit-only analytic ray scaling of fixed bounded increment correction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import collect,correction,observed_score,rollout
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def protected_scale(target,delta,p,fit):
    # Minimize E[(target-a*delta)^2] on fit rows only; a=0 remains feasible.
    numerator=float((p[fit]*(target[fit]*delta[fit]).mean(-1)).sum())
    denominator=float((p[fit]*(delta[fit]**2).mean(-1)).sum())
    alpha=float(np.clip(numerator/denominator,0,1)) if denominator>0 else 0.
    return alpha,dict(numerator=numerator,denominator=denominator,
                      quadratic_change=(alpha*alpha*denominator-2*alpha*numerator)/p[fit].sum())


def scale_model(model,alpha):
    if not 0<=alpha<=1:raise ValueError('Scale outside protection interval')
    result={k:v.copy() if hasattr(v,'copy') else v for k,v in model.items()}
    result['coef']=model['coef']*alpha;result['cap']=model['cap']*alpha
    return result


def main():
    root=Path('adaptive_search_results');path=root/'increment_residual_model_increment.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:model={k:z[k].copy() for k in z.files}
    s=AdaptiveBeam();data=collect(s);h=data['history'];q=data['q'];n=len(q)
    hh=np.repeat(h,8,axis=0);qq=np.repeat(q,8);rr=np.tile(np.arange(8),n)
    centers=s.base.execute_rule(hh,qq,rr).reshape(n,8,2)
    delta=correction(model,s.base,hh,qq,rr).reshape(n,8,2)
    target=np.angle(np.exp(1j*(data['truth'][:,None]-centers)))
    fit=np.isin(data['video'],SPLITS['train'][:-3])
    alpha,selection=protected_scale(target,delta,data['probability'],fit)
    protected=scale_model(model,alpha)
    np.testing.assert_allclose(correction(protected,s.base,hh,qq,rr).reshape(n,8,2),
                               alpha*delta,rtol=1e-12,atol=1e-15)
    assert selection['quadratic_change']<=1e-15
    np.savez(root/'protected_residual_model.npz',**protected)
    diagnostics={name:{part:observed_score(s,data,centers,m,mask)
                       for part,mask in [('fit',fit),('hold',~fit)]}
                 for name,m in [('zero',None),('full',model),('protected',protected)]}
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    free={'zero':[],'protected':[]}
    for seed in [42017,42018,42019,42020]:
        for name,m in [('zero',None),('protected',protected)]:
            r,p,f=rollout(s,hold,m,seed);free[name].append(r)
            np.savez_compressed(root/f'protected_residual_hold_{name}_{seed}.npz',prediction=p,failed=f,
                                truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,r['objective'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    result=dict(alpha=alpha,fit_selection=selection,diagnostics=diagnostics,holdout=free,
                source_sha256=digest,source_unchanged=True,scaled_output_verified=True,
                mean_objective={k:float(np.mean([r['objective'] for r in v])) for k,v in free.items()},
                paired_objective_difference=[a['objective']-b['objective'] for a,b in zip(free['protected'],free['zero'])],
                note='Single alpha from TRAIN fit10 observed weighted angle-quadratic only, clipped[0,1]. Fixed increment direction. Protection is mean local fit loss, not pointwise, holdout or closed-loop guarantee.3 reused TRAIN hold videos/new4action seeds. No DEV/TEST, no search/check enforced or default promotion.')
    (root/'protected_residual_pilot.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
