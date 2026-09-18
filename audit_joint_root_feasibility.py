"""TRAIN joint-root local minimax and finite-step numerical certificates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from audit_full_root_distribution import mixture_score
from audit_root_video_feasibility import fit_robust


def minimax(linear):
    l=np.asarray(linear,float);videos,roots=l.shape
    if not np.isfinite(l).all():raise ValueError('Nonfinite slopes')
    result=linprog(np.r_[np.zeros(roots),1.],A_ub=np.c_[l,-np.ones(videos)],b_ub=np.zeros(videos),
                   A_eq=np.array([np.r_[np.ones(roots),0.]]),b_eq=[1.],
                   bounds=[(0,None)]*roots+[(None,None)],method='highs')
    if not result.success:raise RuntimeError(result.message)
    # Rebuild feasible simplex vectors, then compute primal/dual bounds directly.
    s=np.maximum(result.x[:-1],0);s/=s.sum()
    weights=np.maximum(-result.ineqlin.marginals,0);weights/=weights.sum()
    upper=float(np.max(l@s));lower=float(np.min(weights@l))
    gap=upper-lower
    if gap < -1e-10 or gap > 1e-8:raise RuntimeError('Primal/dual bound mismatch')
    return dict(direction=s.tolist(),video_weights=weights.tolist(),upper=upper,lower=lower,gap=gap,
                solver_message=result.message,raw_simplex_residual=float(abs(result.x[:-1].sum()-1)),
                raw_inequality_violation=float(max(0.,np.max(l@result.x[:-1]-result.x[-1]))))


def finite_bound(linear,quadratic,weights,limit=.25):
    """For normalized root s, dual-weighted delta >= alpha*(lower+alpha*min(0,eigmin))."""
    weights=np.asarray(weights)
    if (weights<0).any() or not np.isclose(weights.sum(),1.):raise ValueError('Invalid video simplex')
    lower=float(np.min(weights@linear));matrix=np.einsum('v,vij->ij',weights,quadratic)
    eigen=float(np.linalg.eigvalsh((matrix+matrix.T)/2).min())
    bound=lower+limit*min(0.,eigen)
    return dict(linear_lower=lower,weighted_min_eigenvalue=eigen,finite_margin=bound,
                excludes_nonzero_to_limit=bool(bound>1e-10),limit=limit)


def analyze(l,q,videos):
    # l[V,R,H], q[V,H,R,R]
    lp=minimax(l[:,:,0]);s=np.asarray(lp['direction'])
    dl=np.einsum('vrh,r->vh',l,s);dq=np.einsum('r,vhrs,s->vh',s,q,s)
    return dict(videos=videos.tolist(),local=lp,finite_certificate=finite_bound(l[:,:,0],q[:,0],lp['video_weights']),
                direction_short_slopes=dl[:,0].tolist(),direction_fit=fit_robust(dl,dq))


def main():
    source=Path('adaptive_search_results/root_horizon_constraint.json')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    prior=np.asarray(old['prior']);video=np.asarray(old['video']);unique=np.unique(video)
    d=np.eye(8)[None]-prior[:,None,:];matrices=[]
    rng=np.random.default_rng(761)
    for row in old['rows']:
        b=np.asarray(row['horizon_pair_distance']);a=np.asarray(row['horizon_attraction'])
        q=-.5*np.einsum('nri,hnij,nsj->hnrs',d,b,d)
        np.testing.assert_allclose(np.diagonal(q,axis1=-2,axis2=-1).transpose(1,2,0),row['quadratic'],rtol=0,atol=1e-14)
        linear=np.asarray(row['linear'])
        for _ in range(3):
            s=rng.dirichlet(np.ones(8));alpha=float(rng.uniform(0,.25));p=(1-alpha)*prior+alpha*s
            for h in range(3):
                predicted=alpha*(linear[:,:,h]@s)+alpha**2*np.einsum('r,nrs,s->n',s,q[h],s)
                np.testing.assert_allclose(predicted,mixture_score(a[h],b[h],p)-mixture_score(a[h],b[h],prior),rtol=0,atol=1e-14)
        matrices.append(q)
    window_q=np.mean(matrices,0);window_l=np.mean([r['linear'] for r in old['rows']],0)
    l=np.stack([window_l[video==v].mean(0) for v in unique])
    q=np.stack([window_q[:,video==v].mean(1) for v in unique])
    full=analyze(l,q,unique);folds={str(v):analyze(l[unique!=v],q[unique!=v],unique[unique!=v]) for v in unique}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(full=full,folds=folds,video_linear=l.tolist(),video_quadratic=q.tolist(),source_sha256=digest,
                note='CachedTRAIN only. Global w=(1-alpha)prior+alpha*s,s simplex8,alpha<=.25. LP min_s max_video Lvs gives local worstshort slope; feasible primal/dual bounds checked. Finite certificate uses dual-weighted fullQ min eigenvalue and ||s||²<=1, sufficient not necessary; numerical1e-10 margin NOT interval-arithmetic proof. Direction fit only chosenLP direction,not joint nonlinear optimum. Full+10 excludedvideo fitting diagnostics, no evaluation/rollout/late/hold/DEV/TEST/promotion. Source diagonal and random joint fullU identities checked1e-14.')
    Path('adaptive_search_results/joint_root_feasibility.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(full=full,folds={v:dict(local=f['local']['upper'],finite=f['finite_certificate'],alpha=f['direction_fit']['alpha']) for v,f in folds.items()}),indent=2))


if __name__=='__main__':main()
