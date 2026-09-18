"""TRAIN-only horizon-separated root mixture curves and exact 1D constraints."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import candidate_features
from confirm_root_mixture_holdout import collect_moments
from audit_full_root_distribution import change_terms,mixture_score


def optimize_direction(linear,quadratic,constrained=True,limit=.25):
    """Global min of mean horizon quadratic; constrain horizon0 delta<=0."""
    l,q=np.asarray(linear,float),np.asarray(quadratic,float)
    if l.shape!=(3,) or q.shape!=(3,) or not np.isfinite([l,q]).all() or not 0<=limit<=1:
        raise ValueError('Expected finite three-horizon coefficients and limit in [0,1]')
    candidates=[0.,limit]
    if q.mean()>0:candidates.append(float(-l.mean()/(2*q.mean())))
    if constrained and q[0]!=0:candidates.append(float(-l[0]/q[0]))
    eligible=[]
    for alpha in candidates:
        if not 0<=alpha<=limit:continue
        delta=alpha*l+alpha**2*q
        if constrained and delta[0]>1e-12:continue
        eligible.append((float(delta.mean()),alpha,delta))
    value,alpha,delta=min(eligible,key=lambda row:(row[0],row[1]))
    return dict(alpha=alpha,mean_delta=value,horizon_delta=delta.tolist(),short_violation=max(0.,float(delta[0])))


def worker(args):
    w,seed,prior=args
    a,b,terms,guards=collect_moments(AdaptiveBeam(),w,seed,4)
    linear=[];quadratic=[]
    for ta,tb in terms:
        ls=[];qs=[]
        for root in range(8):
            target=np.broadcast_to(np.eye(8)[root],prior.shape)
            l,q=change_terms(ta,tb,target,prior)
            # Quadratic curve must equal direct full-distribution scoring.
            for alpha in [0.,.125,.25]:
                p=(1-alpha)*prior+alpha*target
                np.testing.assert_allclose(mixture_score(ta,tb,p)-mixture_score(ta,tb,prior),alpha*l+alpha**2*q,rtol=0,atol=1e-14)
            ls.append(l);qs.append(q)
        linear.append(np.stack(ls,1));quadratic.append(np.stack(qs,1))
    return dict(seed=seed,attraction=a.tolist(),pair_distance=b.tolist(),guards=guards,
                horizon_attraction=[ta.tolist() for ta,tb in terms],horizon_pair_distance=[tb.tolist() for ta,tb in terms],
                linear=np.stack(linear,-1).tolist(),quadratic=np.stack(quadratic,-1).tolist())


def main():
    directory=Path('adaptive_search_results');source=directory/'full_root_distribution.json'
    paths=[source,Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(source.read_text());engine=AdaptiveBeam()
    w=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    _,prior=candidate_features(engine,w['history']);np.testing.assert_array_equal(prior,old['policies']['zero'])
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(worker,[(w,s,prior) for s in range(451017,451021)])):
            for key in ['attraction','pair_distance','guards']:
                np.testing.assert_array_equal(row[key],old['moments'][i][key])
            np.testing.assert_array_equal(mixture_score(np.asarray(row['attraction']),np.asarray(row['pair_distance']),prior),old['runs']['zero'][i]['window_cost'])
            rows.append(row);print('Exact original root moments restored',row['seed'],flush=True)
    linear=np.mean([r['linear'] for r in rows],axis=(0,1));quadratic=np.mean([r['quadratic'] for r in rows],axis=(0,1))
    directions=[]
    for root in range(8):
        directions.append(dict(root=root,linear=linear[root].tolist(),quadratic=quadratic[root].tolist(),
                               at25=(.25*linear[root]+.25**2*quadratic[root]).tolist(),
                               unconstrained=optimize_direction(linear[root],quadratic[root],False),
                               constrained=optimize_direction(linear[root],quadratic[root],True)))
    best={k:min(directions,key=lambda d:(d[k]['mean_delta'],d[k]['alpha'],d['root']))['root'] for k in ['unconstrained','constrained']}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(directions=directions,best_root=best,rows=rows,prior=prior.tolist(),video=w['video'].tolist(),start=w['start'].tolist(),source_hashes=hashes,
                note='TRAIN-only80roots/original451017-20/P4 exactly replayed. Restore horizon moments. Each direction w=(1-alpha)prior+alpha onehot(r),alpha[0,.25]; exact mean quadratic min by endpoints/stationary/short-constraint root. Short constraint mean TRAIN .5s delta<=0 (roundoff1e-12), NOT eachwindow/video/generalization. Zero always feasible, no contextual model. Reports all8 roots and constrained/unconstrained controls; no late/hold/DEV/TEST or promotion. FiniteP curvature can be negative; no convexity assumption.')
    (directory/'root_horizon_constraint.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(directions=directions,best_root=best),indent=2),flush=True)


if __name__=='__main__':main()
