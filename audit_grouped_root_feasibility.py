"""History-motion two-bin root adjustments: TRAIN-only feasibility."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_joint_root_feasibility import minimax,finite_bound
from audit_root_video_feasibility import fit_robust
from audit_full_root_distribution import mixture_score


def grouped_moments(linear,quadratic,group,video):
    # Inactive windows contribute zero, denominator is ALL windows per video.
    n,r,h=linear.shape; gl=np.zeros((n,2*r,h));gq=np.zeros((n,h,2*r,2*r))
    for b in range(2):
        mask=group==b;sl=slice(b*r,(b+1)*r)
        gl[mask,sl]=linear[mask];gq[mask,:,sl,sl]=quadratic[mask]
    unique=np.unique(video)
    return (np.stack([gl[video==v].mean(0) for v in unique]),
            np.stack([gq[video==v].mean(0) for v in unique]))


def analyze(linear,quadratic,motion,video):
    threshold=float(np.median(motion));group=(motion>=threshold).astype(int)
    l,q=grouped_moments(linear,quadratic,group,video)
    lp=minimax(l[:,:,0]);s=np.asarray(lp['direction']);mass=s.reshape(2,-1).sum(1)
    # Total coefficient mass alpha can be up to .5, but EACH bin at most .25.
    limit=float(.25/mass.max())
    dl=np.einsum('vrh,r->vh',l,s);dq=np.einsum('r,vhrs,s->vh',s,q,s)
    fit=fit_robust(dl,dq,limit=limit)
    return dict(threshold=threshold,counts=np.bincount(group,minlength=2).tolist(),
                per_video_counts={str(v):np.bincount(group[video==v],minlength=2).tolist() for v in np.unique(video)},
                local=lp,finite_certificate=finite_bound(l[:,:,0],q[:,0],lp['video_weights'],limit=.5),
                directional_alpha_limit=limit,bin_direction_mass=mass.tolist(),direction_fit=fit,
                bin_actual_strength=(fit['alpha']*mass).tolist(),
                videos=np.unique(video).tolist())


def main():
    source=Path('adaptive_search_results/root_horizon_constraint.json')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    w=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for key in ['video','start']:np.testing.assert_array_equal(w[key],old[key])
    motion=np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2)))
    prior=np.asarray(old['prior']);d=np.eye(8)[None]-prior[:,None,:]
    b=np.mean([r['horizon_pair_distance'] for r in old['rows']],0)
    a=np.mean([r['horizon_attraction'] for r in old['rows']],0)
    q=(-.5*np.einsum('nri,hnij,nsj->hnrs',d,b,d)).transpose(1,0,2,3)
    l=np.mean([r['linear'] for r in old['rows']],0)
    np.testing.assert_allclose(np.diagonal(q,axis1=-2,axis2=-1).transpose(0,2,1),np.mean([r['quadratic'] for r in old['rows']],0),rtol=0,atol=1e-14)
    # Verify arbitrary two-bin probabilities and per-video weighted quadratic identity.
    group=(motion>=np.median(motion)).astype(int);gl,gq=grouped_moments(l,q,group,w['video'])
    rng=np.random.default_rng(771)
    for _ in range(4):
        z=np.stack([rng.dirichlet(np.ones(8))*rng.uniform(0,.25) for _ in range(2)])
        p=(1-z.sum(1)[group,None])*prior+z[group];flat=z.ravel()
        predicted=np.einsum('vrh,r->vh',gl,flat)+np.einsum('r,vhrs,s->vh',flat,gq,flat)
        direct=np.stack([mixture_score(a[h],b[h],p)-mixture_score(a[h],b[h],prior) for h in range(3)],1)
        direct=np.stack([direct[w['video']==v].mean(0) for v in np.unique(w['video'])])
        np.testing.assert_allclose(predicted,direct,rtol=0,atol=1e-14)
    full=analyze(l,q,motion,w['video'])
    folds={str(v):analyze(l[w['video']!=v],q[w['video']!=v],motion[w['video']!=v],w['video'][w['video']!=v]) for v in np.unique(w['video'])}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(full=full,folds=folds,motion=motion.tolist(),video=w['video'].tolist(),start=w['start'].tolist(),source_sha256=digest,
                note='CachedTRAIN80/4P4 only. Two bins from fit-history RMS median, fold thresholds exclude heldvideo. Per-bin nonnegative root coefficients sum<=.25; total<=.5, hence globaluniform family included. Blockdiagonal quadratic, allwindows/video denominator. LP over normalized16-direction; finite bound totalalpha<=.5 sufficient only. Directional alpha cap .25/max(binmass), NOT unnecessary shared .25cap. Fit only LP direction,not global nonlinearopt. No rollout/evaluation/late/hold/DEV/TEST/promotion; no videoID-based partition.')
    Path('adaptive_search_results/grouped_root_feasibility.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(full=full,folds={v:dict(local=f['local']['upper'],finite=f['finite_certificate']['excludes_nonzero_to_limit'],strength=f['bin_actual_strength']) for v,f in folds.items()}),indent=2))


if __name__=='__main__':main()
