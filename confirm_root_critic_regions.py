"""Frozen four-critic check on both regions of historical adapter holdouts."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_value_coverage import candidate_features
from crossfit_root_mixture_critic import policy_features
from validate_temporal_root_critic import score
from validate_temporal_residual_gate import windows
from confirm_temporal_kernel_holdout import hold_windows
from confirm_root_mixture_holdout import collect_moments
from audit_full_root_distribution import mixture_score


def policies_for(engine,w,models):
    x,p=candidate_features(engine,w['history']); x=policy_features(x,p)
    candidates=np.stack([p]+[.75*p+.25*np.broadcast_to(r,p.shape) for r in np.eye(8)],1)
    scores={k:score(m,x) for k,m in models.items()}
    policies={k:candidates[np.arange(len(p)),s.argmin(1)] for k,s in scores.items()}
    return p,scores,policies


def worker(args):
    region,w,seed,prior,policies,*options=args
    particles=options[0] if options else 16
    a,b,terms,guards=collect_moments(AdaptiveBeam(),w,seed,particles)
    baseline=mixture_score(a,b,prior); results={}
    for name,p in policies.items():
        results[name]=dict(delta=(mixture_score(a,b,p)-baseline).tolist(),
                           horizon_delta=[float((mixture_score(ta,tb,p)-mixture_score(ta,tb,prior)).mean()) for ta,tb in terms])
    return dict(region=region,seed=seed,baseline=baseline.tolist(),results=results,guards=guards,
                attraction=a.tolist(),pair_distance=b.tolist())


def summarize(rows,video):
    out=dict(baseline=float(np.mean([r['baseline'] for r in rows])),strategies={})
    for name in rows[0]['results']:
        delta=np.array([r['results'][name]['delta'] for r in rows]); seeds=delta.mean(1)
        out['strategies'][name]=dict(delta=float(delta.mean()),seed_delta=seeds.tolist(),
              conditional_seed_se=float(seeds.std(ddof=1)/np.sqrt(len(seeds))),better_seeds=int((seeds<0).sum()),
              horizon_delta=np.mean([r['results'][name]['horizon_delta'] for r in rows],0).tolist(),
              per_video_delta={str(v):float(delta[:,video==v].mean()) for v in np.unique(video)})
    return out


def main(confirmation=False):
    root=Path('adaptive_search_results')
    paths=[root/'temporal_root_critic_model.json',root/'temporal_root_critic_evaluation.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen=json.loads(paths[0].read_text()); old=json.loads(paths[1].read_text())
    reference=None
    if confirmation:
        path=root/'root_critic_regions.json'
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        reference=json.loads(path.read_text())
    for p,h in frozen['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    models=frozen['models']; engine=AdaptiveBeam(); late=windows('evaluation')
    for key in ['video','start']:np.testing.assert_array_equal(late[key],old[key])
    prior,scores,policies=policies_for(engine,late,models)
    for name in models:
        np.testing.assert_array_equal(scores[name],old['scores'][name])
        np.testing.assert_array_equal(policies[name],old['policies'][name])
    for row in old['runs']:
        a,b=np.asarray(row['attraction']),np.asarray(row['pair_distance'])
        base=mixture_score(a,b,prior)
        np.testing.assert_array_equal(base,row['baseline'])
        for name,p in policies.items():
            np.testing.assert_array_equal(mixture_score(a,b,p)-base,row['results'][name]['delta'])
    print('Old late policies/scores/all cached scores exact',flush=True)
    jobs=[];metadata={}
    for region in ['prefix','tail']:
        w=hold_windows(region)
        assert not set(w['video']) & set(late['video'])
        prior,scores,policies=policies_for(engine,w,models)
        metadata[region]=dict(video=w['video'].tolist(),start=w['start'].tolist(),prior=prior.tolist(),
                              scores={k:v.tolist() for k,v in scores.items()},policies={k:v.tolist() for k,v in policies.items()})
        if reference is not None:
            assert metadata[region]==reference['metadata'][region]
            for row in reference['runs']:
                if row['region']!=region:continue
                a,b=np.asarray(row['attraction']),np.asarray(row['pair_distance'])
                base=mixture_score(a,b,prior)
                np.testing.assert_array_equal(base,row['baseline'])
                for name,p in policies.items():
                    np.testing.assert_array_equal(mixture_score(a,b,p)-base,row['results'][name]['delta'])
        seeds=range(721017,721025) if confirmation else range(711017,711021)
        jobs.extend((region,w,seed,prior,policies,32 if confirmation else 16) for seed in seeds)
    print('All frozen policies checked; fixed budget',len(jobs),'region/seed jobs',flush=True)
    runs=[]
    with ProcessPoolExecutor(max_workers=8 if confirmation else 4) as pool:
        for row in pool.map(worker,jobs):
            runs.append(row)
            print(row['region'],row['seed'],{k:float(np.mean(v['delta'])) for k,v in row['results'].items()},'guards',sum(row['guards']),flush=True)
    summary={region:summarize([r for r in runs if r['region']==region],np.asarray(metadata[region]['video'])) for region in metadata}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,metadata=metadata,source_hashes=hashes,
                note='Frozen4 root critics, no fits/tuning. Historical adapter hold18/16/13 prefix/tail24each, fixed711017-20/P16 perroot/300. All policies share root-integrated moments; crosscomponent same particle indices excluded. No DEV/TEST or promotion. Backbone TRAIN and historically reused hold NOT project blind. Tail first history can cross midpoint; targets secondhalf, withinregion overlap. SE conditional on fixed windows, NOT independent videos. Old late policies and all cached scores exact.')
    if confirmation:
        report['note']=report['note'].replace('fixed711017-20/P16','fixed721017-24/P32')+' Old four-seed report preserved; metadata/policies/all cached old scores exact. Increased sampling budget, NOT new video evidence. No sequential seed extension.'
    output='root_critic_regions_confirmation.json' if confirmation else 'root_critic_regions.json'
    (root/output).write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
