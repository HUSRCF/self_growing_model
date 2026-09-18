"""Refit only kernel widths on the existing purged training block."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from validate_temporal_residual_gate import windows
from validate_soft_gates_new_windows import causal_features
from fit_energy_output_kernel import mean_statistics
from spectral_kernel_energy import spectral_cost, value_gradient
from crossfit_soft_kernel_gate import coefficients


def raw_objective(theta, stats):
    anchored, gradient = value_gradient(stats, theta)
    offset = spectral_cost(stats, np.ones_like(stats['coeff']))-stats['baseline']
    return float((anchored+offset)[0]), gradient[0]


def fit_width(points, truth, failed):
    stats = mean_statistics(points, truth, failed, 1025)
    result = minimize(raw_objective,np.log([10.,10.]),args=(stats,),jac=True,method='L-BFGS-B',
                      bounds=[(np.log(1e-4),np.log(1e6))]*2,
                      options=dict(maxiter=200,gtol=1e-7,ftol=1e-12))
    zero = float(stats['baseline'][0])
    return dict(kappa=np.exp(result.x).tolist(), train_cost=float(result.fun), zero_cost=zero,
                use_zero=bool(result.fun>=zero), success=bool(result.success), message=str(result.message),
                gradient=result.jac.tolist(), iterations=int(result.nit),
                at_boundary=((result.x<np.log(1e-4)+1e-6)|(result.x>np.log(1e6)-1e-6)).tolist())


def main():
    root=Path('adaptive_search_results')
    paths=[root/'full_soft_gate_model.json',root/'full_soft_gate_training_terms.json',
           root/'temporal_residual_training.json',root/'temporal_residual_model.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(paths[0].read_text())['model']['kernel']
    engine=AdaptiveBeam();prefix=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8)
    early=windows('train'); batches=[]; motions=[]
    for w,seed,source in [(prefix,482017,json.loads(paths[1].read_text())),(early,621017,json.loads(paths[2].read_text()))]:
        np.testing.assert_array_equal(w['video'],source['video']);np.testing.assert_array_equal(w['start'],source['start'])
        p,f=continuation(engine,w['history'],seed,particles=32)
        if f.any():raise RuntimeError('Training guard; do not exclude samples')
        points,truth,failed=p[:,:,[49,99,299]],w['truth'][:,[49,99,299]],f[:,:,[49,99,299]]
        np.testing.assert_array_equal(score_endpoints(points,truth,failed),source['coefficients']['4097']['baseline'])
        batches.append((points,truth,failed));motions.append(causal_features(engine,w)[0])
    points,truth,failed=[np.concatenate([b[i] for b in batches]) for i in range(3)]
    motion=np.concatenate(motions);bins=(motion>=old['threshold']).astype(int)
    model=dict(threshold=old['threshold'],counts=np.bincount(bins,minlength=2).tolist(),conditional=[[],[]])
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs=[(b,t,pool.submit(fit_width,points[bins==b,:,t],truth[bins==b,t],failed[bins==b,:,t]))
              for b in range(2) for t in range(3)]
        for b,t,job in jobs:
            row=job.result();model['conditional'][b].append(row);print('width',b,t,row,flush=True)
    all_converged=all(g['success'] for group in model['conditional'] for g in group)
    path=root/'temporal_kernel_model.json'
    path.write_text(json.dumps(dict(model=model,source_hashes=hashes,all_converged=all_converged,
        note='120sameTRAIN windows/trajectories as temporal gate pilot. Threshold fixed from original80,ONLY six2D widths refit. '
             '1025 RAW smoothed U objective,exact point fallback,one log10start,bounds1e-4..1e6,maxiter200. '
             'Raw versus anchored differs by theta-independent offset; unchanged gradient. No late evaluation loaded.'),indent=2))
    if not all_converged:
        print('Nonconvergence preserved; no audit/evaluation',flush=True);return
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(coefficients,model,motion[i:i+15],points[i:i+15],truth[i:i+15],failed[i:i+15]) for i in range(0,120,15)]
        chunks=[j.result() for j in jobs]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(chunks=[dict(coefficients=r,precision=g) for r,g in chunks],source_hashes=hashes,
                all_precision_gates_pass=all(g['cost_pass'] and g['gradient_pass'] for _,g in chunks),
                note='All120 training states uniform-alpha2049/4097 gates1e-4/1e-5. This gradient is alpha,NOT log-kappa. '
                     'No refit on fine grids; no late evaluation/hold/DEV/TEST; old failures untouched.')
    (root/'temporal_kernel_training_audit.json').write_text(json.dumps(report,indent=2))
    print('training precision',report['all_precision_gates_pass'],flush=True)


if __name__=='__main__':main()
