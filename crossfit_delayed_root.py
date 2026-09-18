"""Video-excluded delayed intervention, retaining exact first50 outputs."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_delayed_root import collect_delayed
from crossfit_root_constraint import select
from audit_full_root_distribution import mixture_score


def fit_folds(l,q,video):
    return {str(v):select(l[video!=v].mean(0),q[video!=v].mean(0),False) for v in np.unique(video)}


def policies(folds,video):
    n=len(video);base=np.broadcast_to(np.eye(9)[0],(n,9)).copy();learned=base.copy()
    for v in np.unique(video):
        m=folds[str(v)];use=video==v
        learned[use]=(1-m['alpha'])*base[use]+m['alpha']*np.eye(9)[m['root']+1]
    return dict(zero=base,fixed6=.75*base+.25*np.eye(9)[7],crossfit=learned)


def evaluate(args):
    w,seed,p=args;row=collect_delayed(w,seed,16)
    a,b=np.asarray(row['horizon_attraction']),np.asarray(row['horizon_pair_distance'])
    costs={name:np.stack([mixture_score(a[h],b[h],weight) for h in range(3)],1) for name,weight in p.items()}
    error=max(float(np.max(abs(c[:,0]-costs['zero'][:,0]))) for c in costs.values())
    assert error<=1e-14
    # Pathwise first50 equality checked by collect_delayed; enforce exact score identity.
    for c in costs.values():c[:,0]=costs['zero'][:,0]
    return dict(seed=seed,costs={k:v.tolist() for k,v in costs.items()},guards=row['guards'],
                short_score_roundoff=error,short_coefficient_roundoff=row['short_coefficient_roundoff'])


def main():
    directory=Path('adaptive_search_results');source=directory/'delayed_root_training.json'
    paths=[source,Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    w=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    l=np.mean([r['linear'] for r in old['rows']],0);q=np.mean([r['quadratic'] for r in old['rows']],0)
    full=select(l.mean(0),q.mean(0),False)
    assert full['root']==old['best']['root']
    np.testing.assert_allclose(full['alpha'],old['best']['fit']['alpha'],rtol=0,atol=1e-14)
    folds=fit_folds(l,q,w['video']);p=policies(folds,w['video'])
    path=directory/'crossfit_delayed_root_model.json'
    path.write_text(json.dumps(dict(folds=folds,policies={k:v.tolist() for k,v in p.items()},video=w['video'].tolist(),start=w['start'].tolist(),source_hashes=hashes),indent=2))
    replay=json.loads(path.read_text())
    for k,value in policies(replay['folds'],w['video']).items():np.testing.assert_array_equal(value,p[k])
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('Frozen delayed folds',folds,flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(w,s,p) for s in range(771017,771021)]):
            rows.append(row)
            print(row['seed'],{k:float(np.mean(np.asarray(v)-row['costs']['zero'])) for k,v in row['costs'].items()},'guards',sum(row['guards']),flush=True)
    base=np.asarray([r['costs']['zero'] for r in rows]);summary={}
    for name in p:
        d=np.asarray([r['costs'][name] for r in rows])-base
        summary[name]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                           seed_horizon_delta=d.mean(1).tolist(),
                           per_video_horizon={str(v):d[:,w['video']==v].mean((0,1)).tolist() for v in np.unique(w['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,baseline=float(base.mean()),rows=rows,source_hashes=hashes,
                note='Delayed-specific LOVO roots/alpha on old761017-20/P8 labels,9video72fit perfold. All frozen before new771017-20/P16per9components/300. Zero,fullTRAINchosenfixedr6.25,andcrossfit retained. Fixed6 is NOT independent of heldvideo training labels;only crossfit is adapter-video-excluded. First50 fullpaths exact eachcomponent; shortscore roundoff <=1e-14 verified then set structurally exact baseline. Backbone TRAIN/samehistorical80states,NOT projectblind. No timing sweep/late/hold/DEV/TEST/promotion.')
    (directory/'crossfit_delayed_root_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
