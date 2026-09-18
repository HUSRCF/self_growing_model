"""Freeze full-training root critics, then test purged late windows."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import candidate_features
from crossfit_root_mixture_critic import targets, policy_features, worker
from validate_root_mixture_holdout import fit_critic, predict
from validate_temporal_residual_gate import windows
from audit_full_root_distribution import mixture_score


def fit(x, y, videos, contextual):
    if contextual:
        return dict(contextual=True, arrays={k:v.tolist() for k,v in fit_critic(x,y).items()})
    centered = y-y.mean(1,keepdims=True)
    return dict(contextual=False, scores=np.mean([centered[videos==v].mean(0) for v in np.unique(videos)],0).tolist())


def score(model, x):
    if model['contextual']:
        return predict({k:np.asarray(v) for k,v in model['arrays'].items()},x)
    return np.broadcast_to(model['scores'],x.shape[:2]).copy()


def main():
    root=Path('adaptive_search_results')
    paths=[root/'full_root_distribution.json',root/'root_mixture_critic_model.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(paths[0].read_text()); previous=json.loads(paths[1].read_text())
    engine=AdaptiveBeam(); train=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for key in ['video','start']:
        np.testing.assert_array_equal(train[key],old[key])
        np.testing.assert_array_equal(train[key],previous[key])
    x,prior=candidate_features(engine,train['history']); xx=policy_features(x,prior)
    np.testing.assert_array_equal(prior,old['policies']['zero'])
    labels=[]
    for i,m in enumerate(old['moments']):
        a,b=np.asarray(m['attraction']),np.asarray(m['pair_distance'])
        np.testing.assert_array_equal(mixture_score(a,b,prior),old['runs']['zero'][i]['window_cost'])
        l,f,_=targets(a,b,prior); labels.append([l,f])
    labels=np.mean(labels,0); models={}
    for j,target in enumerate(['linear','full']):
        for contextual in [False,True]:
            name=target+('_context' if contextual else '_action')
            replay=np.empty_like(labels[j])
            for video in np.unique(train['video']):
                use=train['video']!=video
                fold=fit(xx[use],labels[j][use],train['video'][use],contextual)
                replay[~use]=score(fold,xx[~use])
            np.testing.assert_array_equal(replay,previous['models'][name]['scores'])
            models[name]=fit(xx,labels[j],train['video'],contextual)
            np.testing.assert_array_equal(score(models[name],xx),score(json.loads(json.dumps(models[name])),xx))
    path=root/'temporal_root_critic_model.json'
    path.write_text(json.dumps(dict(models=models,source_hashes=hashes,note='Same original80TRAIN/4P4streams, four full fits after exact LOVO replay; frozen BEFORE loading late evaluation. No hyperparameter changes.'),indent=2))
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('All folds and serialization exact; four full models frozen',flush=True)
    late=windows('evaluation')
    for v in np.unique(train['video']):
        assert train['start'][train['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    lx,lp=candidate_features(engine,late['history']); lx=policy_features(lx,lp)
    candidates=np.stack([lp]+[.75*lp+.25*np.broadcast_to(r,lp.shape) for r in np.eye(8)],1)
    scores={k:score(m,lx) for k,m in models.items()}
    choices={k:s.argmin(1) for k,s in scores.items()}
    policies={k:candidates[np.arange(len(lp)),c] for k,c in choices.items()}
    runs=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for seed,(a,b,terms,guards) in pool.map(worker,[(late,s) for s in [701017,701018]]):
            baseline=mixture_score(a,b,lp); results={}
            for name,p in policies.items():
                delta=mixture_score(a,b,p)-baseline
                results[name]=dict(delta=delta.tolist(),horizon_delta=[float((mixture_score(ta,tb,p)-mixture_score(ta,tb,lp)).mean()) for ta,tb in terms])
            runs.append(dict(seed=seed,baseline=baseline.tolist(),results=results,guards=guards,attraction=a.tolist(),pair_distance=b.tolist()))
            print(seed,{k:float(np.mean(v['delta'])) for k,v in results.items()},'guards',sum(guards),flush=True)
    summary={}
    for name in models:
        d=np.array([r['results'][name]['delta'] for r in runs])
        summary[name]=dict(delta=float(d.mean()),seed_delta=d.mean(1).tolist(),
                           horizon_delta=np.mean([r['results'][name]['horizon_delta'] for r in runs],0).tolist(),
                           per_video_delta={str(v):float(d[:,late['video']==v].mean()) for v in np.unique(late['video'])},
                           choice_counts=np.bincount(choices[name],minlength=9).tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,baseline=float(np.mean([r['baseline'] for r in runs])),runs=runs,source_hashes=hashes,
                video=late['video'].tolist(),start=late['start'].tolist(),scores={k:v.tolist() for k,v in scores.items()},
                policies={k:v.tolist() for k,v in policies.items()},
                note='40 late windows same10TRAINvideos, histories wholly finalquarter and disjoint from critic-fit windows/targets. New701017/18 P8perroot/300; all4 retained. Reused historical late states NOT project blind; backbone TRAIN. No hold/DEV/TEST/tuning/repeated control/promotion.')
    (root/'temporal_root_critic_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
