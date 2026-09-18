"""TRAIN delayed destination intervention, preserving first50 outputs exactly."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_full_root_distribution import mixture_terms,change_terms,mixture_score
from confirm_root_mixture_holdout import collect_moments
from audit_root_horizon_constraint import optimize_direction


def collect_delayed(w,seed,particles=8):
    engine=AdaptiveBeam();p0,f0=continuation(engine,w['history'],seed,particles=particles)
    endpoints=[p0[:,:,[49,99,299]]];failures=[f0[:,:,[49,99,299]]];guards=[int(f0.any(-1).sum())]
    for root in range(8):
        p,f=continuation(engine,w['history'],seed,particles=particles,root=root,forced_step=50)
        np.testing.assert_array_equal(p[:,:,:50],p0[:,:,:50]);np.testing.assert_array_equal(f[:,:,:50],f0[:,:,:50])
        endpoints.append(p[:,:,[49,99,299]]);failures.append(f[:,:,[49,99,299]]);guards.append(int(f.any(-1).sum()))
    x,dead=embedding(np.stack(endpoints,1)),np.stack(failures,1)
    terms=[mixture_terms(x[:,:,:,h],embedding(w['truth'][:,t-1]),dead[:,:,:,h]) for h,t in enumerate([50,100,300])]
    n=len(w['history']);prior=np.broadcast_to(np.eye(9)[0],(n,9))
    ls=[];qs=[]
    for h,(a,b) in enumerate(terms):
        ll=[];qq=[]
        for root in range(8):
            target=np.broadcast_to(np.eye(9)[root+1],(n,9));l,q=change_terms(a,b,target,prior)
            for alpha in [0.,.125,.25]:
                mix=(1-alpha)*prior+alpha*target
                np.testing.assert_allclose(mixture_score(a,b,mix)-mixture_score(a,b,prior),alpha*l+alpha**2*q,rtol=0,atol=1e-14)
            ll.append(l);qq.append(q)
        ls.append(np.stack(ll,1));qs.append(np.stack(qq,1))
    l,q=np.stack(ls,-1),np.stack(qs,-1)
    error=float(max(np.max(abs(l[:,:,0])),np.max(abs(q[:,:,0]))))
    assert error<=1e-14
    # Exact pathwise equality proved above; remove only numerical cancellation at .5s.
    l[:,:,0]=0.;q[:,:,0]=0.
    return dict(seed=seed,guards=guards,short_coefficient_roundoff=error,linear=l.tolist(),quadratic=q.tolist(),
                horizon_attraction=[a.tolist() for a,b in terms],horizon_pair_distance=[b.tolist() for a,b in terms])


def worker(args):return collect_delayed(*args)


def main():
    root=Path('adaptive_search_results');source=root/'full_root_distribution.json'
    paths=[source,Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};old=json.loads(source.read_text())
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    # Historical default root-at-zero behavior must remain unchanged after API extension.
    a,b,_,guards=collect_moments(engine,w,451017,4)
    np.testing.assert_array_equal(a,old['moments'][0]['attraction'])
    np.testing.assert_array_equal(b,old['moments'][0]['pair_distance']);assert guards==old['moments'][0]['guards']
    print('Historical root-at-zero moments exact',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(worker,[(w,s,8) for s in range(761017,761021)]):
            rows.append(row);print('Delayed TRAIN',row['seed'],'guards',sum(row['guards']),'short roundoff',row['short_coefficient_roundoff'],flush=True)
    l=np.mean([r['linear'] for r in rows],axis=(0,1));q=np.mean([r['quadratic'] for r in rows],axis=(0,1))
    directions=[dict(root=r,linear=l[r].tolist(),quadratic=q[r].tolist(),fit=optimize_direction(l[r],q[r],False)) for r in range(8)]
    best=min(directions,key=lambda d:(d['fit']['mean_delta'],d['fit']['alpha'],d['root']))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(directions=directions,best=best,rows=rows,video=w['video'].tolist(),start=w['start'].tolist(),source_hashes=hashes,
                note='TRAIN80 roots/new761017-20/P8 per9components:unmodified baseline plus8forceddestination components at zero-based50 (51st generated point). First50 entire trajectories/failures exact versus paired baseline. Original event/r uniforms still drawn,hidden inherited without reset; one intervention only. Global Bernoulli mixture original vs forced continuation alpha<=.25; exact component integration with independent-index crosspairs,not prior q-probability weights at forecast origin. .5 coefficients exact structural0 after <=1e-14 cancellation check. New labels generated on original rollout intervention states,not reused root weights. No fitting-side CV/evaluation/late/hold/DEV/TEST/promotion yet; trainopt not performance claim.')
    (root/'delayed_root_training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(directions=directions,best=best),indent=2),flush=True)


if __name__=='__main__':main()
