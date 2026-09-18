"""Independent-RNG evaluation of frozen history-bin constrained root policies."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import candidate_features
from crossfit_root_constraint import probabilities,evaluate


def grouped_probability(folds,motion,video,prior):
    p=prior.copy()
    for v in np.unique(video):
        model=folds[str(v)];use=video==v
        group=(motion[use]>=model['threshold']).astype(int)
        z=model['direction_fit']['alpha']*np.asarray(model['local']['direction']).reshape(2,-1)
        if (z<0).any() or (z.sum(1)>.25+1e-12).any():raise ValueError('Invalid grouped coefficients')
        p[use]=(1-z.sum(1)[group,None])*prior[use]+z[group]
    return p


def main():
    root=Path('adaptive_search_results')
    paths=[root/'grouped_root_feasibility.json',root/'crossfit_root_constraint_model.json',
           Path('v20_rnn_mixture/models/gru_1901.npz'),Path('v20_rnn_mixture/models/frozen_dynamics.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    group=json.loads(paths[0].read_text());old=json.loads(paths[1].read_text())
    assert hashlib.sha256((root/'root_horizon_constraint.json').read_bytes()).hexdigest()==group['source_sha256']
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:
        np.testing.assert_array_equal(w[k],old[k]);np.testing.assert_array_equal(w[k],group[k])
    motion=np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2)))
    np.testing.assert_array_equal(motion,group['motion'])
    _,prior=candidate_features(engine,w['history']);policies=probabilities(old['folds'],w['video'],prior)
    for k,p in policies.items():np.testing.assert_array_equal(p,old['policies'][k])
    policies['grouped']=grouped_probability(group['folds'],motion,w['video'],prior)
    for p in policies.values():
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-14);assert (p>=0).all()
    frozen=dict(folds=group['folds'],policies={k:p.tolist() for k,p in policies.items()},source_hashes=hashes)
    path=root/'crossfit_grouped_root_model.json';path.write_text(json.dumps(frozen,indent=2))
    replay=json.loads(path.read_text())
    np.testing.assert_array_equal(grouped_probability(replay['folds'],motion,w['video'],prior),policies['grouped'])
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('Frozen five policies, old controls exact',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(w,s,policies) for s in range(741017,741021)]):
            rows.append(row)
            print(row['seed'],{k:float(np.mean(np.asarray(v)-row['costs']['zero'])) for k,v in row['costs'].items()},'guards',sum(row['guards']),flush=True)
    base=np.asarray([r['costs']['zero'] for r in rows]);summary={}
    for name in policies:
        d=np.asarray([r['costs'][name] for r in rows])-base
        summary[name]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                           seed_horizon_delta=d.mean(1).tolist(),
                           per_video_horizon={str(v):d[:,w['video']==v].mean((0,1)).tolist() for v in np.unique(w['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,baseline=float(base.mean()),rows=rows,video=w['video'].tolist(),start=w['start'].tolist(),source_hashes=hashes,
                note='Frozen fold-specific TRAINmotionmedian2bins/minimaxdirection/shortpervideo feasible alpha. Eachfold excludes full heldvideo from threshold and labels. New741017-20/P16perroot/300,all5controls shared; no tune or reselection. Same80 historicalTRAINstates/backboneTRAIN,adapterCV only; no late/hold/DEV/TEST/promotion.')
    (root/'crossfit_grouped_root_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
