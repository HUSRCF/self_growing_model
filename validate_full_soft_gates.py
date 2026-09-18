"""Fixed full-TRAIN gate refit, then frozen prefix AND tail holdout comparison."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from training_window_sampler import TrainingPrefixPool
from validate_soft_gates_new_windows import causal_features
from crossfit_soft_kernel_gate import coefficients
from soft_output_mixture import value_gradient,interval_grid_difference
from soft_kernel_gate import fit_gate as fit_motion,predict_gate as predict_motion
from feature_soft_kernel_gate import fit_mapping,design,fit_gate,predict_gate
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import tail_windows


def weights(model,motion,raw):
    n=len(motion);x=design(model['mapping'],raw)
    return dict(zero=np.zeros((n,3)),full=np.ones((n,3)),
                scalar=np.broadcast_to([g['scalar'] for g in model['motion_gates']],(n,3)).copy(),
                contextual=np.stack([predict_motion(g,np.log(motion+1e-12)) for g in model['motion_gates']],1),
                features=np.stack([predict_gate(g,x) for g in model['feature_gates']],1))


def evaluate_region(seed,region,model):
    if region=='prefix':w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    elif region=='tail':
        full=tail_windows('train',300,8);mask=np.isin(full['video'],SPLITS['train'][-3:])
        w={key:value[mask] for key,value in full.items()}
    else:raise ValueError('Unknown region')
    engine=AdaptiveBeam();motion,raw=causal_features(engine,w)
    # Predictions depend only on observed history and frozen parameters, before rollout/truth scoring.
    alpha=weights(model,motion,raw)
    p,f=continuation(engine,w['history'],seed,particles=64)
    results,precision=coefficients(model['kernel'],motion,p[:,:,[49,99,299]],w['truth'][:,[49,99,299]],f[:,:,[49,99,299]])
    terms={key:np.asarray(value) for key,value in results['4097'].items()}
    return dict(seed=seed,region=region,video=w['video'].tolist(),start=w['start'].tolist(),
                coefficients=results,precision=precision,alpha={k:v.tolist() for k,v in alpha.items()},
                costs={name:value_gradient(terms,a)[0].tolist() for name,a in alpha.items()},guards=int(f.any(-1).sum()))


def main():
    root=Path('adaptive_search_results')
    files=[root/'conditional_output_kernel_model.json',root/'feature_soft_gate_models.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source=json.loads(files[0].read_text());assert source['all_converged'];kernel=source['model']
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8)
    assert set(w['video'])==set(kernel['fit_videos'])
    motion,raw=causal_features(engine,w)
    p,f=continuation(engine,w['history'],482017,particles=32)
    points,truth,failed=p[:,:,[49,99,299]],w['truth'][:,[49,99,299]],f[:,:,[49,99,299]]
    old=json.loads((root/'energy_kernel_refinement.json').read_text())
    np.testing.assert_array_equal(w['video'],old['video']);np.testing.assert_array_equal(w['start'],old['start'])
    np.testing.assert_array_equal(score_endpoints(points,truth,failed),old['results']['4097']['baseline'])
    if failed.any():raise RuntimeError('Training guard; no sample exclusion')
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(coefficients,kernel,motion[i:i+20],points[i:i+20],truth[i:i+20],failed[i:i+20]) for i in range(0,80,20)]
        chunks=[job.result()[0] for job in jobs]
    results={str(size):{key:np.concatenate([chunk[str(size)][key] for chunk in chunks])
                       for key in ['baseline','linear','quadratic']} for size in [2049,4097]}
    bounds=interval_grid_difference(results['2049'],results['4097'])
    precision=dict(cost=float(bounds['cost'].max()),gradient=float(bounds['gradient'].max()),
                   cost_pass=bool(bounds['cost'].max()<=1e-4),gradient_pass=bool(bounds['gradient'].max()<=1e-5))
    coefficient_report=dict(coefficients={size:{k:v.tolist() for k,v in r.items()} for size,r in results.items()},
                            precision=precision,video=w['video'].tolist(),start=w['start'].tolist())
    (root/'full_soft_gate_training_terms.json').write_text(json.dumps(coefficient_report,indent=2))
    print('training coefficient precision',precision,flush=True)
    if not (precision['cost_pass'] and precision['gradient_pass']):return
    mapping=fit_mapping(raw);x=design(mapping,raw);terms=results['4097']
    motion_gates=[fit_motion(np.log(motion+1e-12),terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    feature_gates=[fit_gate(x,terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    assert all(a['scalar']==b['scalar'] for a,b in zip(motion_gates,feature_gates))
    model=dict(kernel=kernel,mapping=mapping,motion_gates=motion_gates,feature_gates=feature_gates)
    replay=json.loads(json.dumps(model))
    for name,a in weights(model,motion,raw).items():np.testing.assert_array_equal(a,weights(replay,motion,raw)[name])
    frozen=dict(model=model,source_hashes=dict(hashes),train_precision=precision,
                note='OriginalTRAIN10/80/P32,existing fullconditional kernel unchanged. Exactpoint LQ trainonly; '
                     'fixed61raw/8projectionseed1901 richgate vs motion/scalar/full/zero;same regularization/bounds/fallback. '
                     'Frozen before loading either holdout region;no hyperparameter search.')
    model_path=root/'full_soft_gate_model.json';model_path.write_text(json.dumps(frozen,indent=2))
    hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('all full gates frozen',dict(motion=[g['success'] for g in motion_gates],features=[g['success'] for g in feature_gates]),flush=True)
    runs=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(evaluate_region,seed,region,model) for seed in range(601017,601021) for region in ['prefix','tail']]
        for job in as_completed(jobs):
            row=job.result();runs.append(row);print('done',row['region'],row['seed'],row['precision'],flush=True)
    runs.sort(key=lambda r:(r['region'],r['seed']));summary={}
    for region in ['prefix','tail']:
        selected=[r for r in runs if r['region']==region];video=np.array(selected[0]['video'])
        values={name:np.array([r['costs'][name] for r in selected]) for name in ['zero','full','scalar','contextual','features']}
        result=dict(zero=float(values['zero'].mean()))
        for name in ['full','scalar','contextual','features']:
            delta=values[name]-values['zero'];means=delta.mean((1,2))
            result[name]=dict(score=float(values[name].mean()),delta=float(delta.mean()),seed_deltas=means.tolist(),
                              conditional_seed_se=float(means.std(ddof=1)/2),better_seeds=int((means<0).sum()),
                              horizon_delta=delta.mean((0,1)).tolist(),per_video_delta={str(v):float(delta[:,video==v].mean()) for v in np.unique(video)})
        result['features_minus_scalar']=float((values['features']-values['scalar']).mean());summary[region]=result
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,train_precision=precision,
                all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in runs),
                note='All5strategies frozen,new601017-20/P64,hold18/16/13prefixANDtail24each. Exactpointallalpha2049/4097gates1e-4/1e-5. '
                     'Historically reusedhold/backboneTRAIN,not globalblindtest;seedSE fixed-window only. '
                     'No hold tuning/DEV/TEST/promotion;old anchored-formula precision failure not overwritten.')
    (root/'full_soft_gate_holdout.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,gates=report['all_precision_gates_pass']),indent=2),flush=True)


if __name__=='__main__':main()
