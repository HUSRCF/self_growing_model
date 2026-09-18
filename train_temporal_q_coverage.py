"""Matched Q label budget: prefix-only vs prefix/early-tail initial windows."""
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
from crossfit_conditional_influence import state_features,fit_folds
from validate_temporal_root_critic import score
from validate_temporal_residual_gate import windows
from residual_soft_gate import block_starts
from v20_rnn_mixture.engine.data import load_video
from evaluate_conditional_policy import evaluate,summarize


def prefix_half(video):
    selected=[]
    for v in np.unique(video):
        ids=np.flatnonzero(video==v)
        if len(ids)!=8:raise ValueError('Expected eight prefix windows/video')
        selected.extend(ids[:4])
    return np.asarray(sorted(selected))


def assert_train_boundary(w):
    for v in np.unique(w['video']):
        boundary=3*len(load_video(int(v)))//4
        if (w['start'][w['video']==v]+300>=boundary).any():raise ValueError('Training target reaches late region')


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'conditional_state_coverage.json',directory/'matched_state_coverage_model.json']
    old,previous=[json.loads(p.read_text()) for p in paths]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,h in previous['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();prefix=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(prefix[k],old[k])
    early=windows('train');assert_train_boundary(prefix);assert_train_boundary(early)
    # Validate future evaluation boundary using metadata only, not late targets.
    for v in np.unique(early['video']):
        earliest_late=block_starts(len(load_video(int(v))),'evaluation').min()-31
        assert early['start'][early['video']==v].max()+300<earliest_late
    snap={50:None};continuation(s,prefix['history'],761017,particles=8,snapshots=snap)
    px=[];py=[]
    for j,g in enumerate(old['groups']):
        state={k:snap[50][k][j::8] for k in ['history','q','hidden','failed']}
        for k,h in g['state_hashes'].items():assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
        x,prior=state_features(s,state);np.testing.assert_array_equal(prior,g['prior'])
        px.append(x);py.append(np.asarray(g['rows'][j]['labels']).mean(-1))
    px,py=np.asarray(px),np.asarray(py)
    models={};projection=previous['projection'];training_scores={}
    # Prefix-only control refit must exactly reproduce prior multi-state models.
    for context,kind in [(False,'action'),(True,'context')]:
        folds,pred=fit_folds(np.concatenate(px),np.concatenate(py),np.tile(prefix['video'],4),context)
        np.testing.assert_array_equal(pred,previous['training_scores']['multi_'+kind])
        for m in folds.values():
            if context:
                np.testing.assert_array_equal(m['arrays']['projection'],projection);del m['arrays']['projection']
        assert folds==previous['models']['multi_'+kind]
        models['prefix_'+kind]=folds;training_scores['prefix_'+kind]=pred.tolist()
    snap={50:None};continuation(s,early['history'],761017,particles=8,snapshots=snap)
    states=[{k:snap[50][k][j::8].copy() for k in ['history','q','hidden','failed']} for j in range(4)]
    ex=[]
    for state in states:
        assert not state['failed'].any();ex.append(state_features(s,state)[0])
    ref,rf=continuation(s,early['history'],811000,particles=16)
    reference=embedding(ref[:,:,[99,299]]);truth=embedding(early['truth'][:,[99,299]])
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(worker,[(states[j],truth,reference,811017+j) for j in range(4)]))
    print('Early TRAIN labels complete; guards',sum(sum(r['guards']) for r in rows),flush=True)
    selected=prefix_half(prefix['video']);xx=[];yy=[];vv=[]
    for j in range(4):
        xx.extend([px[j,selected],ex[j]]);yy.extend([py[j,selected],np.asarray(rows[j]['labels']).mean(-1)])
        vv.extend([prefix['video'][selected],early['video']])
    x,y,v=np.concatenate(xx),np.concatenate(yy),np.concatenate(vv)
    assert len(x)==len(np.concatenate(px))==320
    for context,kind in [(False,'action'),(True,'context')]:
        folds,pred=fit_folds(x,y,v,context)
        for m in folds.values():
            if context:
                np.testing.assert_array_equal(m['arrays']['projection'],projection);del m['arrays']['projection']
        models['mixed_'+kind]=folds;training_scores['mixed_'+kind]=pred.tolist()
    labels_path=directory/'temporal_q_coverage_training.json'
    labels_path.write_text(json.dumps(dict(rows=rows,video=early['video'].tolist(),start=early['start'].tolist(),
        prefix_selected_indices=selected.tolist(),reference_guards=int(rf.any(-1).sum()),source_hashes=hashes.copy()),indent=2))
    hashes[str(labels_path)]=hashlib.sha256(labels_path.read_bytes()).hexdigest()
    frozen=dict(models=models,projection=projection,training_scores=training_scores,source_hashes=hashes.copy(),
                note='80 initial windows each:prefix8/video vs prefix first4/video+earlytail4/video. Four actual t50states/window,diagonal811017+j/P4,16branches/action/window, same reference budget P16. SameLOVO128random/ridge.01N;320rows each,action/context2x2. All trainingtargets before3/4boundary; no late targets loaded before model freeze.')
    model_path=directory/'temporal_q_coverage_model.json';model_path.write_text(json.dumps(frozen,indent=2))
    model=json.loads(model_path.read_text());hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    for name,folds in model['models'].items():
        fx,fv=(np.concatenate(px),np.tile(prefix['video'],4)) if name.startswith('prefix') else (x,v)
        for vid,m in folds.items():
            full=dict(m,arrays=dict(m['arrays'],projection=projection)) if m['contextual'] else m
            use=fv==int(vid);np.testing.assert_array_equal(score(full,fx[use]),np.asarray(training_scores[name])[use])
    print('All four models frozen; old prefix fits and serialization exact',flush=True)
    # Only now load late evaluation targets. Keep prefix and late as separate regions.
    late=windows('evaluation');regions={'prefix':prefix,'late':late};evaluation={}
    jobs=[(name,seed) for name in regions for seed in range(851017,851021)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (name,seed),row in zip(jobs,pool.map(evaluate,[(regions[name],model,seed,16) for name,seed in jobs])):
            evaluation.setdefault(name,[]).append(row);print('Temporal coverage',name,seed,'guards',row['guards'],flush=True)
    summaries={}
    for name,rs in evaluation.items():
        summary=summarize(rs,regions[name]['video'],('mixed_context','prefix_context'))
        summary['mixed_minus_prefix_context']=summary.pop('context_minus_action')
        summary['mixed_minus_prefix_context_seed']=summary.pop('context_minus_action_seed');summaries[name]=summary
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summaries,rows=evaluation,source_hashes=hashes,
                note=frozen['note']+' Frozen then new851017-20/P16 full finite25%t50 mixture on prefix80 and late40,all5policies. No held-driven tuning/hold3/DEV/TEST/defaultpromotion; historicallate reused notblind,backboneTRAIN.')
    (directory/'temporal_q_coverage_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:dict(policies={p:dict(delta=a['delta'],horizon=a['horizon_delta'],seeds=a['seed_delta']) for p,a in r['policies'].items()},mixed_minus_prefix_context=r['mixed_minus_prefix_context']) for k,r in summaries.items()},indent=2),flush=True)


if __name__=='__main__':main()
