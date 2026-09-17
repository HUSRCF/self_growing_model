"""Diagnostic calibration of frozen velocity proxy residuals; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_pilot import prefix_sequence, initializer_data, load_block
from velocity_memory_compat import LegacyFeatureBridge
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def residual_stats(residual, weights, covariance):
    r=np.asarray(residual); w=np.asarray(weights,dtype=float)
    if len(r)==0:return None
    w=w/w.sum();mean=w@r; centered=r-mean
    actual=(centered*w[:,None]).T@centered
    inv=np.linalg.inv(covariance)
    distance=np.einsum('ni,ij,nj->n',r,inv,r)
    return dict(n=len(r),mean=mean.tolist(),covariance=actual.tolist(),
        rmse=float(np.sqrt(w@(r*r).sum(1)/2)),
        mean_mahalanobis_squared=float(w@distance),
        mean_bias_mahalanobis_squared=float(mean@inv@mean),
        coverage={str(p):float(w@(distance<=-2*np.log(1-p))) for p in [.5,.9,.95,.99]},
        tail_second_moment_fraction=float(w@(distance*(distance>-2*np.log(.01)))/(w@distance)) if w@distance else 0.)


def causal_groups(base,y):
    t=np.arange(32,len(y)-4)
    h=np.stack([y[i-31:i+1] for i in t])
    q=base.state_from_history(h)[0]
    speed=np.linalg.norm(y[t]-y[t-1],axis=1)
    return q,speed


def main():
    root=Path('adaptive_search_results');model=root/'prefix_velocity_memory_model.npz'
    source=root/'initial_velocity_distribution.json'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,source]}
    cov=np.array(json.loads(source.read_text())['covariance'])
    init=load_block(model)['initializer'];base=LegacyFeatureBridge(AdaptiveBeam().base)
    datasets={}
    for name,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
        rows=[]
        for video in videos:
            y=prefix_sequence(load_video(video));raw,target,last=initializer_data(base,[(video,y)])
            residual=target-(last+raw/init['scale']@init['coef'])
            q,speed=causal_groups(base,y)
            assert len(residual)==len(q)==len(speed)
            rows.append(dict(residual=residual,q=q,speed=speed,
                video=np.full(len(q),video),weight=np.full(len(q),1/(len(videos)*len(q)))))
        datasets[name]={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
    thresholds=np.quantile(datasets['fit']['speed'],[.25,.5,.75])
    reports={}
    for name,data in datasets.items():
        groups={'all':np.ones(len(data['q']),bool)}
        groups.update({f'q{q}':data['q']==q for q in range(8)})
        bucket=np.searchsorted(thresholds,data['speed'],side='right')
        groups.update({f'speed{i}':bucket==i for i in range(4)})
        groups.update({f'video{v}':data['video']==v for v in np.unique(data['video'])})
        reports[name]={}
        for label,mask in groups.items():
            stats=residual_stats(data['residual'][mask],data['weight'][mask],cov)
            if stats is not None:stats.update(videos=np.unique(data['video'][mask]).tolist(),
                                              population_mass=float(data['weight'][mask].sum()))
            reports[name][label]=stats
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(reports=reports,speed_thresholds=thresholds.tolist(),covariance=cov.tolist(),
        source_hashes=hashes,sources_unchanged=True,
        note='Frozen final initializer, observed TRAIN prefixes only. Fit residuals in-sample/optimistic; hold reused by previous research and backbone. Speed thresholds pooled fit frames only; equal-video population weights conditioned on each group. q diagnosed from causal history, not generated register. Ellipse coverage uses exact chi-square2 thresholds for zero-mean Gaussian proxy residuals, not rollout sin/cos marginal coverage. No fitting/rollout/DEV/TEST.')
    (root/'velocity_calibration_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['all'] for k,v in reports.items()},indent=2))


if __name__=='__main__':main()
