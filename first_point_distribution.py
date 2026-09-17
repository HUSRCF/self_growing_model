"""Original frozen F with a single continuous first-generated-point draw."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from validate_multistep_initializer import paired_summary
from v20_rnn_mixture.engine.common import SPLITS,ROOT


def draw_noise(covariance,n,seed):
    values,vectors=np.linalg.eigh(covariance)
    if values.min()<-1e-12:raise ValueError('covariance must be PSD')
    factor=vectors@np.diag(np.sqrt(np.maximum(values,0)))
    return np.random.default_rng(seed).standard_normal((n,2))@factor.T


def main():
    root=Path('adaptive_search_results');s=AdaptiveBeam()
    sources=[ROOT/'models/frozen_dynamics.json',ROOT/'models/gru_1901.npz']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    fit=prefix_windows(SPLITS['train'][:-3],steps=1,per_video=256);n=len(fit['history'])
    first=np.zeros(2);second=np.zeros((2,2))
    for start in range(0,n,128):
        h=fit['history'][start:start+128];truth=fit['truth'][start:start+128,0];q,m=s.machine.initialize(h)
        pe,T,_=s.machine.read(h,q,m);prob=np.einsum('ne,ner->nr',pe,T)
        p=s.base.execute_rule(np.repeat(h,8,axis=0),np.repeat(q,8),np.tile(np.arange(8),len(h))).reshape(len(h),8,2)
        residual=np.angle(np.exp(1j*(truth[:,None]-p)));weight=prob/n
        first+=np.einsum('nr,nrd->d',weight,residual)
        second+=np.einsum('nr,nri,nrj->ij',weight,residual,residual)
    cov=second-np.outer(first,first);cov=(cov+cov.T)/2
    hold=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    _,p,f=rollout(s,hold,None,211017,first_noise=np.zeros((len(hold['history'])*8,2)))
    with np.load(root/'validate_multistep_original_211017.npz') as old:
        np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
    runs={'original':[],'first_point':[]}
    for seed in range(221017,221021):
        noise=draw_noise(cov,len(hold['history'])*8,seed+1000000)
        for name,perturbation in [('original',None),('first_point',noise)]:
            r,p,f=rollout(s,hold,None,seed,first_noise=perturbation);runs[name].append(r)
            np.savez_compressed(root/f'first_point_distribution_{name}_{seed}.npz',prediction=p,failed=f,
                truth=hold['truth'],video=hold['video'],window_start=hold['start'])
            print(seed,name,r['objective'],flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(fit_videos=SPLITS['train'][:-3],fit_windows=n,covariance=cov.tolist(),residual_mean_not_applied=first.tolist(),
        runs=runs,mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        comparison=paired_summary([a['objective']-b['objective'] for a,b in zip(runs['first_point'],runs['original'])]),
        source_hashes=hashes,sources_unchanged=True,zero_noise_replay_exact=True,
        note='Only first generated F point perturbed by independent zero-mean Gaussian, fixed covariance scale1 from probability-weighted TRAIN10prefix residuals, observed history unchanged. No later injections/weak writer/mean shift. Residual covariance is not a calibrated conditional posterior or mixture-likelihood optimum. Original failure semantics preserved. TRAINhold24/P8/300 four fresh seeds, no DEV/TEST/tuning/promotion.')
    (root/'first_point_distribution.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
