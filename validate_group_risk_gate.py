"""Fixed mean versus worst-region objective on cached TRAIN-only new-kernel terms."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from validate_temporal_residual_gate import windows
from validate_soft_gates_new_windows import causal_features
from soft_kernel_gate import predict_gate
from group_risk_gate import fit_gate
from soft_output_mixture import value_gradient,interval_grid_difference


def predictions(models,reference,motion):
    n=len(motion)
    out=dict(mean_scalar=np.broadcast_to([g['scalar'] for g in reference],(n,3)).copy(),
             mean_lbfgs_motion=np.stack([predict_gate(g,np.log(motion+1e-12)) for g in reference],1))
    for mode,gates in models.items():
        out[mode+'_slsqp_motion']=np.stack([predict_gate(g,np.log(motion+1e-12)) for g in gates],1)
        if mode=='worst':out['worst_scalar']=np.broadcast_to([g['scalar'] for g in gates],(n,3)).copy()
    return out


def main():
    root=Path('adaptive_search_results')
    paths=[root/'temporal_kernel_training_audit.json',root/'matched_kernel_gate_model.json',
           root/'full_soft_gate_training_terms.json',root/'temporal_residual_training.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    audit=json.loads(paths[0].read_text());assert audit['all_precision_gates_pass']
    for p,h in audit['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    reference=json.loads(paths[1].read_text())['model']['motion']
    engine=AdaptiveBeam();prefix=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8);early=windows('train')
    sources=[json.loads(p.read_text()) for p in paths[2:]]
    for w,s in zip([prefix,early],sources):
        np.testing.assert_array_equal(w['video'],s['video']);np.testing.assert_array_equal(w['start'],s['start'])
    motion=np.r_[causal_features(engine,prefix)[0],causal_features(engine,early)[0]]
    groups=np.r_[np.zeros(80,int),np.ones(40,int)]
    terms={size:{k:np.concatenate([r['coefficients'][size][k] for r in audit['chunks']])
                 for k in ['baseline','linear','quadratic']} for size in ['2049','4097']}
    np.testing.assert_array_equal(terms['4097']['baseline'],np.concatenate([s['coefficients']['4097']['baseline'] for s in sources]))
    bounds=interval_grid_difference(terms['2049'],terms['4097'])
    assert bounds['cost'].max()<=1e-4 and bounds['gradient'].max()<=1e-5
    fine=terms['4097'];models={mode:[fit_gate(np.log(motion+1e-12),fine['linear'][:,t],fine['quadratic'][:,t],groups,mode)
                                  for t in range(3)] for mode in ['mean','worst']}
    for a,b in zip(models['mean'],reference):assert a['scalar']==b['scalar']
    train={}
    for k,a in predictions(models,reference,motion).items():
        np.testing.assert_array_equal(a,predictions(json.loads(json.dumps(models)),reference,motion)[k])
        delta=value_gradient(fine,a)[0]-fine['baseline']
        train[k]=dict(mean_delta=float(delta.mean()),group_horizon_delta=[delta[groups==g].mean(0).tolist() for g in [0,1]])
    model_path=root/'group_risk_gate_model.json'
    model_path.write_text(json.dumps(dict(models=models,reference=reference,train=train,source_hashes=dict(hashes),
        note='Same120TRAIN/newkernel/causal logmotion;two region risks measured RELATIVE exactpoint. '
             'Mean and worst both SLSQP samebounds/start/L2;original LBFGSmean also retained. '
             'Exact global scalar minimax inclnegativecurvature/crossings;motion local,scalarfallback. Frozen before evaluation read.'),indent=2))
    hashes[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('models frozen', {k:[g['success'] for g in v] for k,v in models.items()},flush=True)
    eval_path=root/'temporal_kernel_evaluation.json';prior=json.loads(eval_path.read_text())
    hashes[str(eval_path)]=hashlib.sha256(eval_path.read_bytes()).hexdigest();assert prior['all_precision_gates_pass']
    for p,h in prior['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    layout_path=root/'matched_kernel_gate_evaluation.json';layout=json.loads(layout_path.read_text())
    hashes[str(layout_path)]=hashlib.sha256(layout_path.read_bytes()).hexdigest()
    late=windows('evaluation');lm=causal_features(engine,late)[0]
    np.testing.assert_array_equal(late['video'],layout['video']);np.testing.assert_array_equal(late['start'],layout['start'])
    for v in np.unique(late['video']):assert early['start'][early['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    alpha=predictions(models,reference,lm);runs=[]
    for row,old in zip(prior['runs'],layout['runs']):
        assert row['seed']==old['seed']
        t={k:np.asarray(v) for k,v in row['coefficients']['4097'].items()}
        costs={k:value_gradient(t,a)[0].tolist() for k,a in alpha.items()}
        np.testing.assert_array_equal(costs['mean_scalar'],old['costs']['new_scalar'])
        np.testing.assert_array_equal(costs['mean_lbfgs_motion'],old['costs']['new_motion'])
        runs.append(dict(seed=row['seed'],costs=costs,baseline=t['baseline'].tolist()))
    baseline=np.asarray([r['baseline'] for r in runs]);summary=dict(zero=float(baseline.mean()))
    for k in alpha:
        values=np.asarray([r['costs'][k] for r in runs]);delta=values-baseline
        summary[k]=dict(score=float(values.mean()),delta=float(delta.mean()),seed_deltas=delta.mean((1,2)).tolist(),
                       horizon_delta=delta.mean((0,1)).tolist(),
                       per_video_delta={str(v):float(delta[:,late['video']==v].mean()) for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,train=train,runs=runs,alpha={k:v.tolist() for k,v in alpha.items()},source_hashes=hashes,
                note='Fixed objective comparison,oldmean predictions exact;cached late40/622017-18/P32. '
                     'No new independent data/hold/DEV/TEST/rollout/integral/weightscan/promotion;precision gates reused.')
    (root/'group_risk_gate_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
