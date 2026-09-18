"""Six-parameter output-center pilot with purged late-block evaluation."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from training_window_sampler import TrainingPrefixPool
from validate_temporal_residual_gate import windows
from output_center_shift import fit_shift,shifted_cost


def rollout(w,seed):
    p,f=continuation(AdaptiveBeam(),w['history'],seed,particles=32)
    return p[:,:,[49,99,299]],f[:,:,[49,99,299]],int(f.any(-1).sum())


def evaluate(w,seed,shifts):
    p,f,guards=rollout(w,seed);truth=w['truth'][:,[49,99,299]]
    zero=score_endpoints(p,truth,f)
    cost=np.stack([shifted_cost(p[:,:,t],truth[:,t],f[:,:,t],shifts[t]) for t in range(3)],1)
    return dict(seed=seed,zero=zero.tolist(),shifted=cost.tolist(),guards=guards)


def main():
    root=Path('adaptive_search_results')
    paths=[root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prefix=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(481017,per_video=8);early=windows('train')
    batches=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(rollout,w,s) for w,s in [(prefix,482017),(early,621017)]]
        for w,path,job in zip([prefix,early],paths[:2],jobs):
            p,f,guards=job.result();old=json.loads(path.read_text());truth=w['truth'][:,[49,99,299]]
            np.testing.assert_array_equal(w['video'],old['video']);np.testing.assert_array_equal(w['start'],old['start'])
            np.testing.assert_array_equal(score_endpoints(p,truth,f),old['coefficients']['4097']['baseline'])
            if guards:raise RuntimeError('Training guard; no exclusion')
            batches.append((p,truth,f))
    points,truth,failed=[np.concatenate([b[i] for b in batches]) for i in range(3)]
    fits=[fit_shift(points[:,:,t],truth[:,t]) for t in range(3)]
    shifts=[g['shift'] for g in fits];train_zero=score_endpoints(points,truth,failed)
    train_delta=np.stack([shifted_cost(points[:,:,t],truth[:,t],failed[:,:,t],shifts[t]) for t in range(3)],1)-train_zero
    model=dict(fits=fits,train_horizon_delta=train_delta.mean(0).tolist(),
               train_prefix_delta=train_delta[:80].mean(0).tolist(),train_early_tail_delta=train_delta[80:].mean(0).tolist(),
               source_hashes=dict(hashes),note='120equalwindows80prefix+40earlytail,P32 oldstreams exact. Six outputshiftparams,not feedback;no newkernel/no width/gate retuning.')
    model_path=root/'output_center_shift_model.json';model_path.write_text(json.dumps(model,indent=2))
    hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('frozen fits',fits,flush=True)
    if not all(g['success'] for g in fits):print('Nonconvergence preserved; no evaluation',flush=True);return
    late=windows('evaluation')
    for v in np.unique(late['video']):assert early['start'][early['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    with ProcessPoolExecutor(max_workers=4) as pool:
        runs=[j.result() for j in [pool.submit(evaluate,late,s,shifts) for s in range(661017,661021)]]
    zero=np.array([r['zero'] for r in runs]);cost=np.array([r['shifted'] for r in runs]);delta=cost-zero;means=delta.mean((1,2))
    summary=dict(zero=float(zero.mean()),shifted=float(cost.mean()),delta=float(delta.mean()),seed_deltas=means.tolist(),
                 conditional_seed_se=float(means.std(ddof=1)/2),better_seeds=int((means<0).sum()),horizon_delta=delta.mean((0,1)).tolist(),
                 per_video_delta={str(v):float(delta[:,late['video']==v].mean()) for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,video=late['video'].tolist(),start=late['start'].tolist(),source_hashes=hashes,
                note='Frozen outputshift vszero on purged late40,new661017-20/P32. No smoothing/integral/hold/DEV/TEST/tuning/promotion. '
                     'Historical fitting videos,not project-blind. No inferred effect on rollout dynamics.')
    (root/'output_center_shift_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
