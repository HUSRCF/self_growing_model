"""Frozen full-training motion-bin root policy on purged late windows."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import candidate_features
from audit_full_root_distribution import mixture_score
from crossfit_grouped_root_constraint import grouped_probability
from crossfit_root_constraint import evaluate
from validate_temporal_residual_gate import windows


def full_probabilities(models,motion,prior):
    video=np.zeros(len(prior),dtype=int)
    out=dict(zero=prior.copy(),fixed5=.75*prior+.25*np.eye(prior.shape[1])[5])
    out['grouped']=grouped_probability({'0':models['grouped']},motion,video,prior)
    for name in ['unconstrained','constrained']:
        m=models[name];out[name]=(1-m['alpha'])*prior+m['alpha']*np.eye(prior.shape[1])[m['root']]
    return out


def main():
    root=Path('adaptive_search_results')
    paths=[root/'grouped_root_feasibility.json',root/'root_horizon_constraint.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    group=json.loads(paths[0].read_text());old=json.loads(paths[1].read_text())
    assert group['source_sha256']==hashes[str(paths[1])]
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    models={'grouped':group['full']}
    for name in ['unconstrained','constrained']:
        r=old['best_root'][name];models[name]=dict(root=r,**old['directions'][r][name])
    engine=AdaptiveBeam();train=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for key in ['video','start']:np.testing.assert_array_equal(train[key],group[key])
    motion=np.sqrt(np.mean(np.diff(train['history'],axis=1)**2,axis=(1,2)))
    np.testing.assert_array_equal(motion,group['motion'])
    _,prior=candidate_features(engine,train['history']);np.testing.assert_array_equal(prior,old['prior'])
    p=full_probabilities(models,motion,prior);deltas=[]
    for row in old['rows']:
        d=np.stack([mixture_score(np.asarray(a),np.asarray(b),p['grouped'])-mixture_score(np.asarray(a),np.asarray(b),prior)
                    for a,b in zip(row['horizon_attraction'],row['horizon_pair_distance'])],1)
        deltas.append(d)
    d=np.mean(deltas,0)
    per_video=np.stack([d[train['video']==v].mean(0) for v in np.unique(train['video'])])
    np.testing.assert_allclose(per_video,group['full']['direction_fit']['per_video_horizon'],rtol=0,atol=1e-14)
    path=root/'temporal_grouped_root_model.json'
    path.write_text(json.dumps(dict(models=models,source_hashes=hashes,note='No fitting: existing fullTRAIN models copied and frozen before loading late windows. Fullgroup training U pervideo/horizon replay1e-14.'),indent=2))
    replay=json.loads(path.read_text())
    for name,value in full_probabilities(replay['models'],motion,prior).items():np.testing.assert_array_equal(value,p[name])
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('Full models frozen; training group U replay and JSON predictions verified',flush=True)
    late=windows('evaluation')
    for v in np.unique(train['video']):
        assert train['start'][train['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    motion=np.sqrt(np.mean(np.diff(late['history'],axis=1)**2,axis=(1,2)))
    _,prior=candidate_features(engine,late['history']);policies=full_probabilities(models,motion,prior)
    counts=np.bincount((motion>=models['grouped']['threshold']).astype(int),minlength=2).tolist()
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(late,s,policies) for s in range(751017,751021)]):
            rows.append(row)
            print(row['seed'],{k:float(np.mean(np.asarray(v)-row['costs']['zero'])) for k,v in row['costs'].items()},'guards',sum(row['guards']),flush=True)
    base=np.asarray([r['costs']['zero'] for r in rows]);summary={}
    for name in policies:
        d=np.asarray([r['costs'][name] for r in rows])-base
        summary[name]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),seed_horizon_delta=d.mean(1).tolist(),
                           per_video_horizon={str(v):d[:,late['video']==v].mean((0,1)).tolist() for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,baseline=float(base.mean()),rows=rows,counts=counts,motion=motion.tolist(),
                video=late['video'].tolist(),start=late['start'].tolist(),policies={k:p.tolist() for k,p in policies.items()},source_hashes=hashes,
                note='FrozenfullTRAIN rootpolicies, temporal late40 same10videos, histories finalquarter and disjoint from rootfit history/targets. New751017-20/P16perroot/300 all5controls retained, no refit/tuning. Historical late states reused/backbone TRAIN,NOT project blind. No hold/DEV/TEST/promotion.')
    (root/'temporal_grouped_root_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(baseline=report['baseline'],counts=counts,summary=summary),indent=2),flush=True)


if __name__=='__main__':main()
