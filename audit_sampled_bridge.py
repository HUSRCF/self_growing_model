"""TRAIN-only long sampled bridge parity; no fitting or gradient-gain claim."""
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from feedback_distribution_pilot import sample
from ensemble_score_objective import energy_costs


def numpy_trace(s,h,steps,seed):
    q,mem=s.machine.initialize(h);rng=np.random.default_rng(seed);dead=np.zeros(len(h),bool);records=[]
    for _ in range(steps):
        pe,tr,read=s.machine.read(h,q,mem)
        e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
        y=s.base.execute_rule(h,q,r)
        dead|=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        y[dead]=h[dead,-1];h=np.concatenate([h[:,1:],y[:,None]],1);q=r;mem={'hidden':read['read_hidden']}
        records.append(dict(event=e.copy(),q=q.copy(),history=h.copy(),hidden=mem['hidden'].copy()))
    return records


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s)
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);results=[]
    for seed in [291017,291018]:
        score,p,f=rollout(s,w,None,seed,particles=8)
        with torch.no_grad():
            actual=sampled_path(b,tensor(np.repeat(w['history'],8,axis=0)),300,tensor([0.,0.]),seed,trace=True)
        pp=actual['prediction'].numpy().reshape(p.shape);ff=actual['failed'].numpy().reshape(f.shape)
        np.testing.assert_array_equal(ff,f)
        reference=numpy_trace(s,np.repeat(w['history'],8,axis=0),300,seed)
        np.testing.assert_array_equal(np.stack([x['history'][:,-1] for x in reference],1).reshape(p.shape),p)
        mismatches={key:sum(int(np.count_nonzero(a[key].numpy()!=r[key])) for a,r in zip(actual['trace'],reference)) for key in ['event','q']}
        errors=np.abs(pp-p)
        costs=[]
        for t in [50,100,300]:
            x=pp[:,:,t-1];y=w['truth'][:,t-1]
            cost,_=energy_costs(np.concatenate([np.sin(x),np.cos(x)],-1),np.c_[np.sin(y),np.cos(y)],ff[:,:,t-1])
            costs.append(cost.mean())
        results.append(dict(seed=seed,max_abs_error=float(errors.max()),
                            original_atol_1e_7_pass=bool((errors<=1e-7).all()),
                            count_above_1e_7=int((errors>1e-7).sum()),action_mismatches=mismatches,
                            max_hidden_error=max(float(np.max(np.abs(a['hidden'].numpy()-r['hidden']))) for a,r in zip(actual['trace'],reference)),
                            horizon_max_errors={str(t):float(errors[:,:,:t].max()) for t in [50,100,300]},
                            failed=int(f.sum()),objective=score['objective'],
                            bridge_objective=float(np.mean(costs)),objective_delta=float(np.mean(costs)-score['objective'])))
        print(results[-1],flush=True)
    report=dict(results=results,windows=24,particles=8,steps=300,
                note='TRAIN-prefix hold only; no fitting/DEV/TEST. Same NumPy uniforms, detached categorical draws. Original 1e-7 gate preserved as reported Boolean, not loosened after failure; traced actions diagnose drift. Synthetic tests cover continuing history/q/hidden after hard failure. Hard-boundary gradients unresolved.')
    Path('adaptive_search_results/sampled_bridge_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
