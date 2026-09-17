"""Observed versus generated-history augmented causal residual training."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import collect, fit_models, observed_score, rollout
from feedback_distribution_pilot import sample
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def generated_rows(s, observed, seed=41017):
    """Truth supplies labels only, never carrier transitions or features."""
    rows={k:[] for k in observed}
    for v in SPLITS['train']:
        indices=np.flatnonzero(observed['video']==v)
        # collect() emits 64 time slices, each with 16 parallel histories.
        blocks=indices.reshape(64,16)
        h=observed['history'][blocks[0]].copy();q,mem=s.machine.initialize(h)
        rng=np.random.default_rng(seed+v)
        for block in blocks:
            pe,tr,read=s.machine.read(h,q,mem)
            values=dict(history=h.copy(),q=q.copy(),truth=observed['truth'][block].copy(),
                        probability=np.einsum('be,ber->br',pe,tr),video=np.full(len(h),v))
            for k in rows:rows[k].append(values[k])
            e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
            y=s.base.execute_rule(h,q,r)
            if not np.isfinite(y).all() or (np.abs(y-h[:,-1])>np.pi).any():
                raise ValueError('Invalid training carrier; do not silently fit held states')
            h=np.concatenate([h[:,1:],y[:,None]],1);q=r;mem={'hidden':read['read_hidden']}
    return {k:np.concatenate(v) for k,v in rows.items()}


def main():
    s=AdaptiveBeam();root=Path('adaptive_search_results')
    observed=collect(s);generated=generated_rows(s,observed)
    mixed={k:np.concatenate([observed[k],generated[k]]) for k in observed}
    original,oc,of=fit_models(s.base,observed)
    augmented,mc,mf=fit_models(s.base,mixed)
    # Same physical correction budget; augmentation must not win by larger cap.
    augmented['ridge_0.0001']['cap']=original['ridge_0.0001']['cap'].copy()
    models={'zero':None,'observed':original['ridge_0.0001'],'mixed':augmented['ridge_0.0001']}
    _,gc,gf=fit_models(s.base,generated)
    diagnostics={}
    for name,model in models.items():
        diagnostics[name]={}
        for label,data,centers,fit in [('observed',observed,oc,of),('generated',generated,gc,gf)]:
            diagnostics[name][label]={part:observed_score(s,data,centers,model,mask)
                                     for part,mask in [('fit',fit),('hold',~fit)]}
        if model is not None:
            np.savez(root/f'drift_residual_model_{name}.npz',**model)
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    free={name:[] for name in models}
    for seed in [32017,32018,32019,32020]:
        for name,model in models.items():
            result,pred,failed=rollout(s,hold,model,seed)
            free[name].append(result)
            np.savez_compressed(root/f'drift_residual_hold_{name}_{seed}.npz',prediction=pred,failed=failed,
                                truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,result['objective'],flush=True)
    report=dict(diagnostics=diagnostics,holdout=free,alpha=.0001,carrier_seed=41017,
                common_cap=models['observed']['cap'].tolist(),observed_rows=len(observed['q']),
                generated_rows=len(generated['q']),
                mean_objective={k:float(np.mean([r['objective'] for r in v])) for k,v in free.items()},
                note='TRAIN prefixes only;10fit/3previously reused hold videos. Fixed alpha and observed-fit cap. Mixed has equally weighted observed/generated states at identical target times, twice rows but no additional independent truth. Normalization follows each fit distribution. No checker enforced, diagnostics only. No DEV/TEST, no hyperparameter sweep; no deployment promotion.')
    (root/'drift_residual_pilot.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
