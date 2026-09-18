"""Causal mid-trajectory Q critics, video-crossfit and fresh reference evaluation."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_conditional_influence import worker
from audit_delayed_state import correlation
from train_search_value import features
from validate_temporal_root_critic import fit, score
from audit_value_coverage import predict_crossfit


def state_features(s, state):
    h,q,hidden=[state[k].copy() for k in ['history','q','hidden']]
    pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
    xx=[]
    for r in range(8):
        rs=np.full(len(h),r);y=s.base.execute_rule(h,q,rs)
        child=np.concatenate([h[:,1:],y[:,None]],1)
        xx.append(np.c_[features(s.base,child,rs,read['read_hidden']),np.eye(8)[q]])
    return np.stack(xx,1),np.einsum('ne,ner->nr',pe,tr)


def fit_folds(x,target,video,contextual):
    folds={};scores=np.empty_like(target)
    for v in np.unique(video):
        use=video!=v
        if not use.any():raise ValueError('Need independent fitting videos')
        model=fit(x[use],target[use],video[use],contextual)
        scores[~use]=score(model,x[~use]);folds[str(v)]=model
    return folds,scores


def main():
    directory=Path('adaptive_search_results');source=directory/'conditional_influence_audit.json'
    old=json.loads(source.read_text());hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    for p,h in old['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    snapshots={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snapshots)
    state={k:snapshots[50][k][::8].copy() for k in ['history','q','hidden','failed']}
    for k,h in old['state_hashes'].items():assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
    x,prior=state_features(s,state);np.testing.assert_array_equal(prior,old['prior'])
    target=np.asarray([r['labels'] for r in old['rows'][:2]]).mean((0,3))
    models={};scores={}
    for contextual,name in [(False,'action'),(True,'context')]:
        models[name],scores[name]=fit_folds(x,target,w['video'],contextual)
        np.testing.assert_array_equal(scores[name],predict_crossfit(x,target,w['video'],x,w['video'],contextual))
    choices={k:v.argmin(1) for k,v in scores.items()}
    choices.update(prior_mode=prior.argmax(1),fixed_r6=np.full(len(prior),6))
    # Store projection once; its fixed seed/map is identical across context folds.
    projection=models['context'][str(w['video'][0])]['arrays']['projection']
    for m in models['context'].values():
        np.testing.assert_array_equal(m['arrays']['projection'],projection)
        del m['arrays']['projection']
    frozen=dict(models=models,projection=projection,scores={k:v.tolist() for k,v in scores.items()},
                choices={k:v.tolist() for k,v in choices.items()},source_hashes=hashes.copy(),
                video=w['video'].tolist(),start=w['start'].tolist(),
                note='Actual fixed t50state, same causal child features/random128/ridge .01N as existing critic. LOVO preprocessing/labels excluded. Fit only801017/18 mean1s/3s influence. All folds frozen before fresh reference/future evaluation; no hyperparameter selection.')
    path=directory/'conditional_influence_critic_model.json';path.write_text(json.dumps(frozen,indent=2))
    serialized=json.loads(path.read_text())
    for name,folds in serialized['models'].items():
        for v,m in folds.items():
            if m['contextual']:m['arrays']['projection']=serialized['projection']
            use=w['video']==int(v)
            np.testing.assert_array_equal(score(m,x[use]),scores[name][use])
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('Two critics frozen; fold/serialization predictions exact',flush=True)
    ref,reff=continuation(s,w['history'],811000,particles=16)
    reference=embedding(ref[:,:,[99,299]]);truth=embedding(w['truth'][:,[99,299]])
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(worker,[(state,truth,reference,seed) for seed in range(811017,811021)]):
            rows.append(row);print('Fresh conditional evaluation',row['seed'],'guards',sum(row['guards']),flush=True)
    y=np.asarray([r['labels'] for r in rows]);base=np.einsum('snrh,nr->snh',y,prior)
    idx=np.arange(len(prior));summary={};deltas={}
    for name,choice in choices.items():
        delta=y[:,idx,choice]-base;deltas[name]=delta
        summary[name]=dict(delta=float(delta.mean()),seed_delta=delta.mean((1,2)).tolist(),
                           horizon_delta=delta.mean((0,1)).tolist(),seed_horizon_delta=delta.mean(1).tolist(),
                           per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])},
                           choice_counts=np.bincount(choice,minlength=8).tolist())
        if name in scores:
            centered=y.mean((0,3));centered-=centered.mean(1,keepdims=True)
            summary[name]['centered_label_correlation']=correlation(scores[name].ravel(),centered.ravel())
    difference=deltas['context']-deltas['action']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,rows=rows,source_hashes=hashes,reference_guards=int(reff.any(-1).sum()),
                context_minus_action=float(difference.mean()),context_minus_action_seed=difference.mean((1,2)).tolist(),
                note=frozen['note']+' Fresh811000/P16 full-prefix reference and811017-20/P4per8roots/250 from SAME first-particle states. Influence slopes only,NOT finite policy energy gain. No recursive feedback/late/hold/DEV/TEST/default promotion. Backbone sawTRAIN; only critic-video-separated; four RNGs not independent videos.')
    (directory/'conditional_influence_critic_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['summary','context_minus_action','context_minus_action_seed']},indent=2),flush=True)


if __name__=='__main__':main()
