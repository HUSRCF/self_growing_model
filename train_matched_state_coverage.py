"""Equal continuation-label budget: repeated state vs diverse states."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from crossfit_conditional_influence import state_features,fit_folds
from validate_temporal_root_critic import score
from evaluate_conditional_policy import evaluate,summarize


def datasets(features,labels,video):
    # labels[group,stream,window,action,horizon], features[group,window,action,D].
    x,y,v=np.asarray(features),np.asarray(labels),np.asarray(video)
    if x.ndim!=4 or y.ndim!=5 or x.shape[0]!=4 or y.shape[:2]!=(4,4) or x.shape[:3]!=(4,len(v),y.shape[3]) or y.shape[2]!=len(v):
        raise ValueError('Expected four groups/four streams and aligned windows/actions')
    return dict(single=(x[0],y[0].mean((0,3)),v.copy()),
                multi=(np.concatenate(x),np.concatenate([y[g,g].mean(-1) for g in range(4)]),np.tile(v,4)))


def main():
    directory=Path('adaptive_search_results');source=directory/'conditional_state_coverage.json'
    old=json.loads(source.read_text());hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    snap={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snap)
    xx=[];labels=[]
    for j,g in enumerate(old['groups']):
        assert g['particle_index']==j
        assert [r['seed'] for r in g['rows']]==list(range(811017,811021))
        state={k:snap[50][k][j::8] for k in ['history','q','hidden','failed']}
        for k,h in g['state_hashes'].items():assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
        x,prior=state_features(s,state);np.testing.assert_array_equal(prior,g['prior'])
        xx.append(x);labels.append([r['labels'] for r in g['rows']])
    data=datasets(xx,labels,w['video']);models={};predictions={};projection=None
    for family,(x,y,v) in data.items():
        for contextual,kind in [(False,'action'),(True,'context')]:
            name=family+'_'+kind;folds,pred=fit_folds(x,y,v,contextual)
            for vid,m in folds.items():
                use=v==int(vid)
                np.testing.assert_array_equal(score(json.loads(json.dumps(m)),x[use]),pred[use])
                if contextual:
                    if projection is None:projection=m['arrays']['projection']
                    np.testing.assert_array_equal(m['arrays']['projection'],projection)
                    del m['arrays']['projection']
            models[name]=folds;predictions[name]=pred.tolist()
    frozen=dict(models=models,projection=projection,training_scores=predictions,source_hashes=hashes.copy(),
                video=w['video'].tolist(),start=w['start'].tolist(),
                note='Same16 continuation branches/window/action:single first-state four811017-20streams×P4, multi fourstates diagonalstream(group0→811017,etc)×P4. Same811000reference,LOVO/random128/ridge.01N,action/context2×2. Single80 vs multi320rows; equal rollout-label budget NOT equal fitting CPU or all historical diagnostic cost. No tuning/diagonal selection.')
    path=directory/'matched_state_coverage_model.json';path.write_text(json.dumps(frozen,indent=2))
    model=json.loads(path.read_text());hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for family,(x,y,v) in data.items():
        for kind in ['action','context']:
            name=family+'_'+kind
            for vid,m in model['models'][name].items():
                full=dict(m,arrays=dict(m['arrays'],projection=model['projection'])) if m['contextual'] else m
                use=v==int(vid);np.testing.assert_array_equal(score(full,x[use]),np.asarray(predictions[name])[use])
    print('All four critics frozen and serialization exact',flush=True)
    # Existing finite evaluator and shared resume API must still replay old results.
    previous_path=directory/'conditional_policy_evaluation.json'
    hashes[str(previous_path)]=hashlib.sha256(previous_path.read_bytes()).hexdigest()
    previous=json.loads(previous_path.read_text())['rows'][0]
    oldmodel=json.loads((directory/'conditional_influence_critic_model.json').read_text())
    replay=evaluate((w,oldmodel,821017,16))
    np.testing.assert_array_equal(replay['baseline'],previous['baseline'])
    for name in replay['mixed']:
        np.testing.assert_array_equal(replay['mixed'][name],previous['mixed'][name])
        for key in ['linear','quadratic']:np.testing.assert_array_equal(replay['coefficients'][name][key],previous['coefficients'][name][key])
    print('Historical finite policies/coefficients exact',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(w,model,seed,16) for seed in range(831017,831021)]):
            rows.append(row);print('Matched coverage',row['seed'],'guards',row['guards'],flush=True)
    summary={}
    for family in ['single','multi']:
        remapped=[dict(r,mixed={kind:r['mixed'][family+'_'+kind] for kind in ['action','context']},
                       coefficients={kind:r['coefficients'][family+'_'+kind] for kind in ['action','context']}) for r in rows]
        summary[family]=summarize(remapped,w['video'])
    comparison={}
    for kind in ['action','context']:
        d=np.asarray([np.asarray(r['mixed']['multi_'+kind])-r['mixed']['single_'+kind] for r in rows])
        comparison[kind]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                              per_video_horizon={str(v):d[:,w['video']==v].mean((0,1)).tolist() for v in np.unique(w['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,multi_minus_single=comparison,rows=rows,source_hashes=hashes,
                note=frozen['note']+' Frozen before NEW831017-20/P16 fullprefix allparticle t50 policies. Full25%continuation mixture U/LQ;first50exact. No late/hold/DEV/TEST/feedbackevery-step/defaultpromotion. All cached labels TRAIN; backboneseenTRAIN,critic-videoCV only.')
    (directory/'matched_state_coverage_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,multi_minus_single=comparison),indent=2),flush=True)


if __name__=='__main__':main()
