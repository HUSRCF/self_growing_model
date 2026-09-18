"""Video-excluded root/strength fitting, then independent RNG evaluation."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import candidate_features
from audit_root_horizon_constraint import optimize_direction
from confirm_root_mixture_holdout import collect_moments
from audit_full_root_distribution import mixture_score


def select(linear,quadratic,constrained):
    choices=[dict(root=r,**optimize_direction(l,q,constrained)) for r,(l,q) in enumerate(zip(linear,quadratic))]
    return min(choices,key=lambda d:(d['mean_delta'],d['alpha'],d['root']))


def fit_folds(linear,quadratic,videos):
    return {str(v):{name:select(linear[videos!=v].mean(0),quadratic[videos!=v].mean(0),flag)
                   for name,flag in [('unconstrained',False),('constrained',True)]} for v in np.unique(videos)}


def probabilities(folds,videos,prior):
    result=dict(zero=prior.copy(),fixed5=.75*prior+.25*np.eye(prior.shape[1])[5])
    for name in ['unconstrained','constrained']:
        p=prior.copy()
        for v in np.unique(videos):
            m=folds[str(v)][name];use=videos==v
            p[use]=(1-m['alpha'])*prior[use]+m['alpha']*np.eye(prior.shape[1])[m['root']]
        result[name]=p
    return result


def evaluate(args):
    w,seed,policies=args
    a,b,terms,guards=collect_moments(AdaptiveBeam(),w,seed,16)
    costs={k:np.stack([mixture_score(ta,tb,p) for ta,tb in terms],1).tolist() for k,p in policies.items()}
    return dict(seed=seed,costs=costs,guards=guards)


def main():
    directory=Path('adaptive_search_results');source=directory/'root_horizon_constraint.json'
    paths=[source,Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    _,prior=candidate_features(engine,w['history']);np.testing.assert_array_equal(prior,old['prior'])
    l=np.mean([r['linear'] for r in old['rows']],0);q=np.mean([r['quadratic'] for r in old['rows']],0)
    # Match original full fit before any video exclusions.
    for name,flag in [('unconstrained',False),('constrained',True)]:
        m=select(l.mean(0),q.mean(0),flag)
        assert m['root']==old['best_root'][name]
        np.testing.assert_allclose(m['alpha'],old['directions'][m['root']][name]['alpha'],rtol=0,atol=1e-14)
    folds=fit_folds(l,q,w['video']); policies=probabilities(folds,w['video'],prior)
    frozen=dict(folds=folds,policies={k:p.tolist() for k,p in policies.items()},video=w['video'].tolist(),start=w['start'].tolist(),source_hashes=hashes)
    path=directory/'crossfit_root_constraint_model.json';path.write_text(json.dumps(frozen,indent=2))
    restored=json.loads(path.read_text())
    for k,p in probabilities(restored['folds'],w['video'],prior).items():np.testing.assert_array_equal(p,policies[k])
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('All video-excluded root/alpha policies frozen',folds,flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(evaluate,[(w,s,policies) for s in range(731017,731021)]):
            rows.append(row)
            print(row['seed'],{k:float(np.mean(np.asarray(v)-row['costs']['zero'])) for k,v in row['costs'].items()},'guards',sum(row['guards']),flush=True)
    base=np.asarray([r['costs']['zero'] for r in rows]); summary={}
    for name in policies:
        d=np.asarray([r['costs'][name] for r in rows])-base
        summary[name]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                           per_video_delta={str(v):float(d[:,w['video']==v].mean()) for v in np.unique(w['video'])},
                           per_video_horizon={str(v):d[:,w['video']==v].mean((0,1)).tolist() for v in np.unique(w['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,baseline=float(base.mean()),rows=rows,source_hashes=hashes,
                note='All10 LOVO folds9videos72windows:fit root/alpha only original451017-20/P4 labels; short TRAIN mean<=0 not pervideo guarantee. Frozen before new731017-20/P16perroot/300. Prior,fixed5/.25,unconstrained,constrained all retained. Adapter-only CV; backbone TRAIN/samehistorical80states not project blind. No late/hold/DEV/TEST/tuning/promotion.')
    (directory/'crossfit_root_constraint_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
