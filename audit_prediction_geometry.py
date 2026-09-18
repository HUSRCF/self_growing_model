"""TRAIN-only endpoint geometry: exact embedding decomposition and circular residuals."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from ensemble_score_objective import energy_costs
from training_window_sampler import TrainingPrefixPool
from validate_temporal_residual_gate import windows


def correlation(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def geometry(points,truth,failed):
    x,y=embedding(points),embedding(truth)
    center=x.mean(1);spread=((x-center[:,None])**2).sum(-1).mean(1)
    center_error=((center-y)**2).sum(-1)
    center_by_angle=(center[:,:2]-y[:,:2])**2+(center[:,2:]-y[:,2:])**2
    variance=((x-center[:,None])**2).mean(1)
    spread_by_angle=variance[:,:2]+variance[:,2:]
    particle_error=((x-y[:,None])**2).sum(-1).mean(1)
    np.testing.assert_allclose(center_error+spread,particle_error,rtol=1e-13,atol=1e-13)
    distance=np.linalg.norm(x-y[:,None],axis=-1)
    pair=np.linalg.norm(x[:,:,None]-x[:,None,:],axis=-1)
    diversity=pair.sum((1,2))/(2*points.shape[1]*(points.shape[1]-1))
    score=distance.mean(1)-diversity+2*failed.mean(1)
    np.testing.assert_array_equal(score,energy_costs(x,y,failed)[0])
    circular=np.arctan2(np.sin(points).mean(1),np.cos(points).mean(1))
    residual=np.arctan2(np.sin(truth-circular),np.cos(truth-circular))
    resultants=np.sqrt(np.sin(points).mean(1)**2+np.cos(points).mean(1)**2)
    sin_noise=np.sin(points-circular[:,None]);sin_noise-=sin_noise.mean(1,keepdims=True)
    return dict(center_error=center_error,spread=spread,particle_error=particle_error,score=score,
                center_error_by_angle=center_by_angle,spread_by_angle=spread_by_angle,
                second_moment_excess=center_by_angle-(points.shape[1]+1)/(points.shape[1]-1)*spread_by_angle,
                attraction=distance.mean(1),diversity=diversity,nearest=distance.min(1),
                residual=residual,resultant=resultants,sin_noise=sin_noise)


def summarize(data,mask):
    d={k:v[mask] for k,v in data.items()}
    residual=d['residual'];noise=d['sin_noise'].reshape(-1,2)
    return dict(count=int(mask.sum()),**{k:float(d[k].mean()) for k in ['score','center_error','spread','particle_error','attraction','diversity','nearest']},
                **{k:d[k].mean(0).tolist() for k in ['center_error_by_angle','spread_by_angle','second_moment_excess']},
                circular_bias=np.arctan2(np.sin(residual).mean(0),np.cos(residual).mean(0)).tolist(),
                residual_sine_correlation=correlation(np.sin(residual[:,0]),np.sin(residual[:,1])),
                ensemble_sine_correlation=correlation(noise[:,0],noise[:,1]),
                resultant_quantiles=np.quantile(d['resultant'],[0,.25,.5,.75,1],axis=0).tolist(),
                low_resultant_count=(d['resultant']<.1).sum(0).tolist(),
                spread_center_error_correlation=correlation(d['spread'],d['center_error']))


def evaluate(region,seed):
    engine=AdaptiveBeam()
    w=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8) if region=='prefix' else windows('train')
    p,f=continuation(engine,w['history'],seed,particles=32)
    out=[];summary=[];per_video={str(v):[] for v in np.unique(w['video'])}
    for t in [49,99,299]:
        d=geometry(p[:,:,t],w['truth'][:,t],f[:,:,t]);summary.append(summarize(d,np.ones(len(w['video']),bool)))
        for v in np.unique(w['video']):per_video[str(v)].append(summarize(d,w['video']==v))
        out.append({k:v.tolist() for k,v in d.items() if k!='sin_noise'})
    return dict(region=region,seed=seed,video=w['video'].tolist(),start=w['start'].tolist(),summary=summary,
                per_video=per_video,windows=out,guards=int(f.any(-1).sum()))


def main():
    root=Path('adaptive_search_results')
    paths=[root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    tasks=[('prefix',482017),('early_tail',621017)]+[(r,s) for r in ['prefix','early_tail'] for s in [651017,651018]]
    with ProcessPoolExecutor(max_workers=6) as pool:runs=list(pool.map(run_task,tasks))
    for row,path in zip(runs[:2],paths[:2]):
        old=json.loads(path.read_text())
        np.testing.assert_array_equal(row['video'],old['video']);np.testing.assert_array_equal(row['start'],old['start'])
        np.testing.assert_array_equal(np.array([w['score'] for w in row['windows']]).T,old['coefficients']['4097']['baseline'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(runs=runs,source_hashes=hashes,note='Fitting10 only:original80prefix+40earlytail,old RNG baseline exact replay plus651017/18/P32. '
        'Embedding squared-error decomposition is exact but NOT a decomposition of U-energy. Circular center weak when resultant low. '
        'Center error is NOT systematic bias. Second-moment excess subtracts (P+1)/(P-1)*population particle variance; '
        'expectation zero only under conditional iid exchangeable truth/particles. Descriptive, cannot separate bias from underdispersion. '
        'Correlations descriptive,particles/windows not independent;center residual correlation does not identify missing kernel covariance. '
        'No fitting/late internal evaluation/hold/DEV/TEST/integral/promotion.')
    (root/'prediction_geometry.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([dict(region=r['region'],seed=r['seed'],summary=r['summary']) for r in runs],indent=2),flush=True)


def run_task(args):return evaluate(*args)


if __name__=='__main__':main()
