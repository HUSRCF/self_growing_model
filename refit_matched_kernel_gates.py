"""Train-only gate recalibration for the new frozen kernel, reusing audited L/Q."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from validate_temporal_residual_gate import windows,new_weights
from validate_soft_gates_new_windows import causal_features
from soft_kernel_gate import fit_gate as fit_motion,predict_gate as motion_predict
from feature_soft_kernel_gate import fit_mapping,design,fit_gate
from residual_soft_gate import fit_gate as fit_residual
from soft_output_mixture import value_gradient,interval_grid_difference


def fit_model(motion,raw,terms):
    mapping=fit_mapping(raw);x=design(mapping,raw)
    mg=[fit_motion(np.log(motion+1e-12),terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    fg=[fit_gate(x,terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    rg=[fit_residual(x,motion_predict(mg[t],np.log(motion+1e-12)),terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    return dict(mapping=mapping,motion=mg,features=fg,residual=rg)


def main():
    root=Path('adaptive_search_results')
    paths=[root/'temporal_kernel_training_audit.json',root/'temporal_kernel_model.json',
           root/'temporal_residual_model.json',root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    audit=json.loads(paths[0].read_text());assert audit['all_precision_gates_pass']
    for p,h in audit['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    old=json.loads(paths[2].read_text())['model']
    prefix_source=json.loads(paths[3].read_text());early_source=json.loads(paths[4].read_text())
    engine=AdaptiveBeam();prefix=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8);early=windows('train')
    for w,s in [(prefix,prefix_source),(early,early_source)]:
        np.testing.assert_array_equal(w['video'],s['video']);np.testing.assert_array_equal(w['start'],s['start'])
    pm,pr=causal_features(engine,prefix);em,er=causal_features(engine,early)
    motion,raw=np.r_[pm,em],np.concatenate([pr,er])
    before={k:np.concatenate([prefix_source['coefficients']['4097'][k],early_source['coefficients']['4097'][k]])
            for k in ['baseline','linear','quadratic']}
    replay=fit_model(motion,raw,before)
    for k,a in new_weights(old,motion,raw).items():np.testing.assert_array_equal(a,new_weights(replay,motion,raw)[k])
    terms={size:{k:np.concatenate([r['coefficients'][size][k] for r in audit['chunks']])
                 for k in before} for size in ['2049','4097']}
    np.testing.assert_array_equal(terms['4097']['baseline'],before['baseline'])
    bounds=interval_grid_difference(terms['2049'],terms['4097'])
    assert bounds['cost'].max()<=1e-4 and bounds['gradient'].max()<=1e-5
    model=fit_model(motion,raw,terms['4097']);roundtrip=json.loads(json.dumps(model))
    for k,a in new_weights(model,motion,raw).items():np.testing.assert_array_equal(a,new_weights(roundtrip,motion,raw)[k])
    train_summary={}
    for k,a in new_weights(model,motion,raw).items():
        change=value_gradient(terms['4097'],a)[0]-value_gradient(terms['4097'],new_weights(old,motion,raw)[k])[0]
        train_summary[k]=dict(refit_change=float(change.mean()),prefix_change=float(change[:80].mean()),
                              early_tail_change=float(change[80:].mean()))
    model_path=root/'matched_kernel_gate_model.json'
    model_path.write_text(json.dumps(dict(model=model,source_hashes=dict(hashes),
        note='Same120trainingfeatures/normalization/specification as temporal residual pilot;only newkernel LQ targets. '
             'Oldfit predictions replayexact;no rollout/integral/late labels in fitting. Frozen before loading evaluation.'),indent=2))
    hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('matched model frozen', {k:[g['success'] for g in model[k]] for k in ['motion','features','residual']},flush=True)
    # Evaluation artifacts first read after all fitting and model serialization.
    eval_path=root/'temporal_kernel_evaluation.json'
    previous=json.loads(eval_path.read_text());hashes[str(eval_path)]=hashlib.sha256(eval_path.read_bytes()).hexdigest()
    assert previous['all_precision_gates_pass']
    for p,h in previous['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    late=windows('evaluation');lm,lr=causal_features(engine,late)
    layout_path=root/'temporal_residual_evaluation.json'
    layout=json.loads(layout_path.read_text())
    hashes[str(layout_path)]=hashlib.sha256(layout_path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(late['video'],layout['video']);np.testing.assert_array_equal(late['start'],layout['start'])
    for v in np.unique(late['video']):assert early['start'][early['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    alpha=new_weights(model,lm,lr);old_alpha=new_weights(old,lm,lr)
    runs=[]
    for row in previous['runs']:
        fine={k:np.asarray(v) for k,v in row['coefficients']['4097'].items()}
        for k,a in old_alpha.items():np.testing.assert_array_equal(value_gradient(fine,a)[0],row['costs'][k])
        runs.append(dict(seed=row['seed'],costs={k:value_gradient(fine,a)[0].tolist() for k,a in alpha.items()}))
    baseline=np.asarray([r['costs']['old_zero'] for r in previous['runs']]);summary=dict(zero=float(baseline.mean()))
    for k in alpha:
        values=np.asarray([r['costs'][k] for r in runs]);old_values=np.asarray([r['costs'][k] for r in previous['runs']])
        delta=values-baseline;change=values-old_values
        summary[k]=dict(score=float(values.mean()),delta=float(delta.mean()),refit_change=float(change.mean()),
                        seed_deltas=delta.mean((1,2)).tolist(),seed_refit_changes=change.mean((1,2)).tolist(),
                        horizon_delta=delta.mean((0,1)).tolist(),
                        per_video_delta={str(v):float(delta[:,late['video']==v].mean()) for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,train_summary=train_summary,runs=runs,source_hashes=hashes,video=late['video'].tolist(),start=late['start'].tolist(),
                alpha={k:a.tolist() for k,a in alpha.items()},
                note='Cached purged late40/622017-18/P32 coefficients;no new independent evaluation. '
                     'Old scores exact replay,allalpha precision gates reused unchanged. '
                     'No hold/DEV/TEST/rollout/integral/hyperparameter search/promotion.')
    (root/'matched_kernel_gate_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
