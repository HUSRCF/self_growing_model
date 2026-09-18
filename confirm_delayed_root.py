"""Two-component equivalent evaluator and fixed eight-stream confirmation."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_full_root_distribution import mixture_terms,mixture_score


def score_pair(points,truth,failed):
    a,b=mixture_terms(points,truth,failed)
    prior=np.broadcast_to([1.,0.],a.shape);mix=np.broadcast_to([.75,.25],a.shape)
    return mixture_score(a,b,prior),mixture_score(a,b,mix)


def evaluate(args):
    w,seed,particles=args;engine=AdaptiveBeam()
    p,f=continuation(engine,w['history'],seed,particles=particles)
    c,cf=continuation(engine,w['history'],seed,particles=particles,root=6,forced_step=50)
    np.testing.assert_array_equal(p[:,:,:50],c[:,:,:50]);np.testing.assert_array_equal(f[:,:,:50],cf[:,:,:50])
    base=[];mixed=[]
    for t in [49,99,299]:
        z,m=score_pair(embedding(np.stack([p[:,:,t],c[:,:,t]],1)),embedding(w['truth'][:,t]),np.stack([f[:,:,t],cf[:,:,t]],1))
        base.append(z);mixed.append(m)
    base=np.stack(base,1);mixed=np.stack(mixed,1)
    error=float(np.max(abs(mixed[:,0]-base[:,0])));assert error<=1e-14
    mixed[:,0]=base[:,0]
    return dict(seed=seed,baseline=base.tolist(),mixed=mixed.tolist(),guards=[int(f.any(-1).sum()),int(cf.any(-1).sum())],short_roundoff=error)


def main():
    root=Path('adaptive_search_results')
    paths=[root/'crossfit_delayed_root_model.json',root/'crossfit_delayed_root_evaluation.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model=json.loads(paths[0].read_text());old=json.loads(paths[1].read_text())
    for p,h in model['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert all(m['root']==6 and m['alpha']==.25 for m in model['folds'].values())
    np.testing.assert_array_equal(model['policies']['fixed6'],model['policies']['crossfit'])
    w=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],model[k])
    with ProcessPoolExecutor(max_workers=4) as pool:
        replay=list(pool.map(evaluate,[(w,r['seed'],16) for r in old['rows']]))
    maximum=0.
    for actual,previous in zip(replay,old['rows']):
        np.testing.assert_array_equal(actual['baseline'],previous['costs']['zero'])
        diff=np.asarray(actual['mixed'])-previous['costs']['crossfit'];maximum=max(maximum,float(np.max(abs(diff))))
        np.testing.assert_allclose(actual['mixed'],previous['costs']['crossfit'],rtol=0,atol=1e-14)
        assert actual['guards']==[previous['guards'][0],previous['guards'][7]]
    print('All four old9-component results matched; max error',maximum,flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(evaluate,[(w,s,32) for s in range(781017,781025)]):
            rows.append(row);print(row['seed'],np.mean(np.asarray(row['mixed'])-row['baseline'],axis=0),flush=True)
    d=np.asarray([np.asarray(r['mixed'])-r['baseline'] for r in rows]);seed=d.mean((1,2));hseed=d.mean(1)
    summary=dict(baseline=float(np.mean([r['baseline'] for r in rows])),delta=float(d.mean()),seed_delta=seed.tolist(),
                 conditional_seed_se=float(seed.std(ddof=1)/np.sqrt(len(seed))),horizon_delta=d.mean((0,1)).tolist(),
                 horizon_conditional_seed_se=(hseed.std(0,ddof=1)/np.sqrt(len(seed))).tolist(),seed_horizon_delta=hseed.tolist(),
                 per_video_horizon={str(v):d[:,w['video']==v].mean((0,1)).tolist() for v in np.unique(w['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,rows=rows,old_replay_max_error=maximum,source_hashes=hashes,
                note='Fixeddelayedr6/.25 atzero-based50,all10LOVOfolds already same; no refit/timing/strength sweep. Same80TRAINstates,new781017-24/P32,fixed8streams. Only baseline+forced6 components;allfour old771017-20/P16ninecomponent scores replay1e-14,baseline exact. First50 paths/failures exact. SE conditional sampling,not newvideo evidence; backboneTRAIN. No sequential seed extension/late/hold/DEV/TEST/promotion.')
    (root/'delayed_root_confirmation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
