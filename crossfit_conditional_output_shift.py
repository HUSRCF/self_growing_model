"""Adapter-level excluded-video motion shift with fresh evaluation RNG."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from validate_temporal_residual_gate import windows
from validate_output_center_shift import rollout
from audit_periodic_output_kernel import score_endpoints
from output_center_shift import fit_shift,attraction_gradient
from conditional_output_shift import fit_model,predict


def main():
    root=Path('adaptive_search_results')
    paths=[root/'output_shift_stability.json',root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original=json.loads(paths[0].read_text())['models']['original']['folds']
    ws=[TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(481017,per_video=8),windows('train')]
    video=np.concatenate([w['video'] for w in ws]);truth=np.concatenate([w['truth'][:,[49,99,299]] for w in ws])
    motion=np.concatenate([np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2))) for w in ws]);log_motion=np.log(motion+1e-12)
    with ProcessPoolExecutor(max_workers=2) as pool:
        batches=[j.result() for j in [pool.submit(rollout,w,s) for w,s in zip(ws,[482017,621017])]]
    for w,path,(p,f,guards) in zip(ws,paths[1:3],batches):
        old=json.loads(path.read_text());np.testing.assert_array_equal(w['video'],old['video']);np.testing.assert_array_equal(w['start'],old['start'])
        np.testing.assert_array_equal(score_endpoints(p,w['truth'][:,[49,99,299]],f),old['coefficients']['4097']['baseline'])
        if guards:raise RuntimeError('TRAIN guard; no exclusion')
    points=np.concatenate([b[0] for b in batches]);models={}
    for v in np.unique(video):
        fit=video!=v
        common=[fit_shift(points[fit,:,t],truth[fit,t]) for t in range(3)]
        np.testing.assert_array_equal([g['shift'] for g in common],[g['shift'] for g in original[str(v)]])
        conditional=[fit_model(log_motion[fit],points[fit,:,t],truth[fit,t]) for t in range(3)]
        for g in conditional:np.testing.assert_array_equal(predict(g,log_motion[~fit]),predict(json.loads(json.dumps(g)),log_motion[~fit]))
        models[str(v)]=dict(common=common,conditional=conditional,fit_videos=np.unique(video[fit]).tolist())
    model_path=root/'conditional_output_shift_models.json'
    model_path.write_text(json.dumps(dict(models=models,source_hashes=dict(hashes),note='Fold9 only norm and targets,108windows;4params/horizon clip-affine logmotion,±.25 outputs/coefs,.001allcoefL2,zero init,TRAIN zero fallback. No evaluation RNG generated before freeze.'),indent=2))
    hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    runs=[]
    for seed in [681017,681018]:
        with ProcessPoolExecutor(max_workers=2) as pool:
            batches=[j.result() for j in [pool.submit(rollout,w,seed) for w in ws]]
        p=np.concatenate([b[0] for b in batches]);f=np.concatenate([b[1] for b in batches]);zero=score_endpoints(p,truth,f)
        delta={k:np.zeros((120,3)) for k in ['common','conditional']}
        for v in np.unique(video):
            ids=video==v;m=models[str(v)]
            for t in range(3):
                baseline=attraction_gradient(p[ids,:,t],truth[ids,t],[0.,0.])[0]
                shifts=dict(common=m['common'][t]['shift'],conditional=predict(m['conditional'][t],log_motion[ids])[:,None,:])
                for name,s in shifts.items():delta[name][ids,t]=attraction_gradient(p[ids,:,t],truth[ids,t],s)[0]-baseline
        runs.append(dict(seed=seed,zero=zero.tolist(),delta={k:v.tolist() for k,v in delta.items()},guards=sum(b[2] for b in batches)))
    summary=dict(zero=float(np.mean([r['zero'] for r in runs])))
    for name in ['common','conditional']:
        d=np.array([r['delta'][name] for r in runs])
        summary[name]=dict(delta=float(d.mean()),seed_deltas=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                           prefix_delta=float(d[:,:80].mean()),early_tail_delta=float(d[:,80:].mean()),
                           per_video_delta={str(v):float(d[:,video==v].mean()) for v in np.unique(video)})
    fits=[g for m in models.values() for g in m['conditional']]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,converged=sum(g['success'] for g in fits),
                selected=sum(g['use_shift'] for g in fits),boundary=sum(any(g['at_boundary']) for g in fits),
                note='New681017/18/P32 same120 fitting-side states;adapter9video excludes evaluation video,backbone allTRAIN. '
                     'Commoncontrol replayexact,conditional4vscommon2params/h. No late/hold/DEV/TEST/tuning/promotion.')
    (root/'conditional_output_shift_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2)[:10000],flush=True)


if __name__=='__main__':main()
