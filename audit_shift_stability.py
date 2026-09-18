"""TRAIN-only shift sensitivity to RNG and excluded video; no late-block access."""
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


def fit_all(points,truth):return [fit_shift(points[:,:,t],truth[:,t]) for t in range(3)]


def delta_for(points,truth,fits):
    return np.stack([attraction_gradient(points[:,:,t],truth[:,t],fits[t]['shift'])[0]-
                     attraction_gradient(points[:,:,t],truth[:,t],[0.,0.])[0] for t in range(3)],1)


def main():
    root=Path('adaptive_search_results')
    paths=[root/'output_center_shift_model.json',root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prefix=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(481017,per_video=8);early=windows('train')
    data_windows=[prefix,early];truth=np.concatenate([w['truth'][:,[49,99,299]] for w in data_windows]);video=np.r_[prefix['video'],early['video']]
    streams={'original':[482017,621017],'rng651017':[651017,651017],'rng651018':[651018,651018]}
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs={name:[pool.submit(rollout,w,s) for w,s in zip(data_windows,seeds)] for name,seeds in streams.items()}
        data={name:[j.result() for j in rows] for name,rows in jobs.items()}
    for w,path,batch in zip(data_windows,paths[1:3],data['original']):
        source=json.loads(path.read_text());np.testing.assert_array_equal(w['video'],source['video']);np.testing.assert_array_equal(w['start'],source['start'])
        np.testing.assert_array_equal(score_endpoints(batch[0],w['truth'][:,[49,99,299]],batch[1]),source['coefficients']['4097']['baseline'])
    guards={name:sum(b[2] for b in rows) for name,rows in data.items()}
    if any(guards.values()):raise RuntimeError('Training guards; no sample exclusion')
    points={name:np.concatenate([b[0] for b in rows]) for name,rows in data.items()}
    models={}
    for name,p in points.items():
        full=fit_all(p,truth)
        folds={str(v):fit_all(p[video!=v],truth[video!=v]) for v in np.unique(video)}
        models[name]=dict(full=full,folds=folds)
    original=json.loads(paths[0].read_text())['fits']
    np.testing.assert_array_equal([f['shift'] for f in models['original']['full']],[f['shift'] for f in original])
    # Freeze all full and excluded-video fits before any cross-stream scoring.
    cases=[]
    for trained,model in models.items():
        for evaluated,p in points.items():
            full_delta=delta_for(p,truth,model['full']);cross_delta=np.zeros((120,3))
            for v in np.unique(video):
                ids=video==v;cross_delta[ids]=delta_for(p[ids],truth[ids],model['folds'][str(v)])
            cases.append(dict(fit_stream=trained,evaluation_stream=evaluated,
                              full_mean=float(full_delta.mean()),full_horizon=full_delta.mean(0).tolist(),
                              excluded_mean=float(cross_delta.mean()),excluded_horizon=cross_delta.mean(0).tolist(),
                              excluded_per_video={str(v):float(cross_delta[video==v].mean()) for v in np.unique(video)},
                              excluded_window_delta=cross_delta.tolist()))
    sensitivity={}
    for name,model in models.items():
        shifts=np.array([[f['shift'] for f in fs] for fs in model['folds'].values()])
        sensitivity[name]=dict(full_shift=[f['shift'] for f in model['full']],
                               excluded_shift_min=shifts.min(0).tolist(),excluded_shift_max=shifts.max(0).tolist(),
                               excluded_shift_std=shifts.std(0).tolist())
    fits=[f for model in models.values() for group in [model['full'],*model['folds'].values()] for f in group]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(models=models,cases=cases,sensitivity=sensitivity,guards=guards,source_hashes=hashes,
                converged=sum(f['success'] for f in fits),total_fits=len(fits),boundary_fits=sum(any(f['at_boundary']) for f in fits),
                note='Only original120 fitting states,3TRAIN streams,P32;99fits samebounds/L2/start. '
                     'Full-model crossRNG shares futuretruth,NOT new labels/generalization. '
                     'Excluded-video adapters omit that video labels but backbone trained allTRAIN;windows overlap. '
                     'No late internal eval/hold/DEV/TEST/hyperparameter search/promotion.')
    (root/'output_shift_stability.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(sensitivity=sensitivity,cases=[{k:v for k,v in r.items() if k not in ['excluded_window_delta','excluded_per_video']} for r in cases],
                         converged=report['converged'],total=report['total_fits'],boundary=report['boundary_fits']),indent=2),flush=True)


if __name__=='__main__':main()
