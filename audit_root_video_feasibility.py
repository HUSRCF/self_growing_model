"""Exact intersection of per-video short-horizon quadratic constraints."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_root_horizon_constraint import optimize_direction


def positive_interval(linear,quadratic,limit=.25):
    """Feasible positive alpha closure; alpha=0 always independently feasible."""
    l,q=np.asarray(linear,float),np.asarray(quadratic,float)
    if l.shape!=q.shape or l.ndim!=1 or not np.isfinite([l,q]).all() or not 0<limit<=1:
        raise ValueError('Expected finite matching vectors and positive limit<=1')
    lo,hi=0.,float(limit)
    for a,b in zip(l,q):
        if b>0:hi=min(hi,float(-a/b))
        elif b<0:lo=max(lo,float(-a/b))
        elif a>0:return None
    return [lo,hi] if hi>0 and lo<=hi else None


def fit_robust(linear,quadratic,limit=.25):
    """[videos,3 horizons]; equal video objective, every video short constraint."""
    l,q=np.asarray(linear),np.asarray(quadratic)
    interval=positive_interval(l[:,0],q[:,0],limit);candidates=[0.]
    ml,mq=l.mean(0),q.mean(0)
    if interval is not None:
        candidates+=interval
        if mq.mean()>0:
            stationary=-ml.mean()/(2*mq.mean())
            if interval[0]<=stationary<=interval[1]:candidates.append(float(stationary))
    alpha=min(candidates,key=lambda a:(float((a*ml+a*a*mq).mean()),a))
    delta=alpha*l+alpha**2*q
    assert delta[:,0].max()<=1e-12
    return dict(alpha=alpha,positive_interval=interval,mean_delta=float(delta.mean()),
                per_video_horizon=delta.tolist(),horizon_delta=delta.mean(0).tolist())


def analyze(l,q,videos):
    directions=[]
    for root in range(l.shape[1]):
        single=[positive_interval([a],[b]) for a,b in zip(l[:,root,0],q[:,root,0])]
        directions.append(dict(root=root,short_linear=l[:,root,0].tolist(),short_quadratic=q[:,root,0].tolist(),
            individual_intervals={str(v):i for v,i in zip(videos,single)},
            mean_constraint=optimize_direction(l[:,root].mean(0),q[:,root].mean(0)),
            robust=fit_robust(l[:,root],q[:,root])))
    best=min(directions,key=lambda d:(d['robust']['mean_delta'],d['robust']['alpha'],d['root']))
    return dict(directions=directions,best_root=best['root'],best=best['robust'],videos=videos.tolist())


def main():
    source=Path('adaptive_search_results/root_horizon_constraint.json')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    video=np.asarray(old['video']);unique=np.unique(video)
    linear=np.mean([r['linear'] for r in old['rows']],0);quadratic=np.mean([r['quadratic'] for r in old['rows']],0)
    l=np.stack([linear[video==v].mean(0) for v in unique]);q=np.stack([quadratic[video==v].mean(0) for v in unique])
    full=analyze(l,q,unique)
    for d,original in zip(full['directions'],old['directions']):
        np.testing.assert_allclose(d['mean_constraint']['alpha'],original['constrained']['alpha'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(d['mean_constraint']['mean_delta'],original['constrained']['mean_delta'],rtol=0,atol=1e-14)
    folds={str(v):analyze(l[unique!=v],q[unique!=v],unique[unique!=v]) for v in unique}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(full=full,folds=folds,video_linear=l.tolist(),video_quadratic=q.tolist(),source_sha256=digest,
                note='Only cached80TRAIN roots/451017-20/P4. For alpha>0, pervideo short delta<=0 equivalent L+alpha Q<=0; intersect linear halfintervals and include alpha0 separately. Handles negativeQ/disconnected positive interval. Equalvideo objective,one fixedroot direction at a time,NOT jointroot mixtures or all possible causal policies. Full+10 excludedvideo TRAIN feasible sets; no evaluation labels/rollouts/late/hold/DEV/TEST/promotion. Meanconstraint replay atol1e-14 due grouping order.')
    Path('adaptive_search_results/root_video_feasibility.json').write_text(json.dumps(report,indent=2))
    print('Full',[(d['root'],d['robust']['positive_interval'],d['robust']['alpha']) for d in full['directions']])
    print('Folds',{v:(f['best_root'],f['best']['alpha']) for v,f in folds.items()})
    print('Individual blocked videos',{d['root']:[v for v,i in d['individual_intervals'].items() if i is None] for d in full['directions']})


if __name__=='__main__':main()
