"""Fixed richer causal gate, fold-only normalization and fresh evaluation RNG."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from train_search_value import features
from audit_crossfit_value import continuation
from crossfit_soft_kernel_gate import evaluate_fold
from feature_soft_kernel_gate import fit_mapping,design,fit_gate,predict_gate
from soft_kernel_gate import predict_gate as motion_predict
from soft_output_mixture import value_gradient


def evaluate(args):
    base_args,model,raw=args
    row=evaluate_fold(base_args)
    x=design(model['mapping'],raw)
    alpha=np.stack([predict_gate(g,x) for g in model['gates']],1)
    terms={key:np.array(value) for key,value in row['coefficients']['4097'].items()}
    row['alpha']['features']=alpha.tolist()
    row['costs']['features']=value_gradient(terms,alpha)[0].tolist()
    return row


def main():
    root=Path('adaptive_search_results')
    files=[root/'soft_kernel_gate_models.json',root/'crossfit_output_kernel_models.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    old=json.loads(files[0].read_text());kernels=json.loads(files[1].read_text())['models']
    assert old['train_precision_pass']
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8)
    motion=np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2)))
    q,memory=engine.machine.initialize(w['history']);_,_,read=engine.machine.read(w['history'],q,memory)
    raw=np.c_[features(engine.base,w['history'],q,read['read_hidden']),np.log(motion+1e-12)]
    models=[]
    for fold,kernel in zip(old['folds'],kernels):
        assert fold['held_video']==kernel['held_video']
        ids=np.flatnonzero(w['video']!=kernel['held_video'])
        np.testing.assert_array_equal(ids,fold['fit_indices'])
        assert set(w['video'][ids])==set(kernel['fit_videos'])
        mapping=fit_mapping(raw[ids]);x=design(mapping,raw[ids])
        terms=fold['coefficients']['4097'];l=np.array(terms['linear']);qq=np.array(terms['quadratic'])
        gates=[fit_gate(x,l[:,t],qq[:,t]) for t in range(3)]
        for gate,control in zip(gates,fold['gates']):assert gate['scalar']==control['scalar']
        models.append(dict(held_video=kernel['held_video'],fit_indices=ids.tolist(),mapping=mapping,gates=gates))
    frozen=dict(models=models,source_hashes=hashes,raw_dimension=raw.shape[1],
                note='Causal continuous history features+sourceq onehot+readhidden+logmotion. Foldtrain-onlymean/std,clip±3, '
                     'fixed RNG1901 Gaussian projection/ sqrtD to8tanhfeatures,intercept;9params/horizon. '
                     'Same .001sumslopes² penalty,bounds[-2,3]/[-2,2],start[.5,0..],LBFGS200,fit-only scalar fallback. '
                     'More capacity than motion2params;not representation-only controlled comparison. All TRAIN LQ reused without refit.')
    path=root/'feature_soft_gate_models.json';path.write_text(json.dumps(frozen,indent=2))
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Replay prior motion-gate predictions without using prior evaluation labels for fitting.
    previous=json.loads((root/'soft_kernel_gate_evaluation.json').read_text())
    for row in previous['runs']:
        fold=next(r for r in old['folds'] if r['held_video']==row['held_video'])
        ids=row['indices'];alpha=np.stack([motion_predict(g,np.log(motion[ids]+1e-12)) for g in fold['gates']],1)
        np.testing.assert_array_equal(alpha,row['alpha']['contextual'])
    print('frozen richer gates, raw dimension',raw.shape[1],'old controls exact',flush=True)
    tasks=[];seeds=[581017,581018];truth=w['truth'][:,[49,99,299]]
    for seed in seeds:
        p,f=continuation(engine,w['history'],seed,particles=32)
        for kernel,fold,model in zip(kernels,old['folds'],models):
            ids=np.flatnonzero(w['video']==kernel['held_video'])
            base=(seed,kernel,fold,ids,motion[ids],p[ids][:,:,[49,99,299]],truth[ids],f[ids][:,:,[49,99,299]])
            tasks.append((base,model,raw[ids]))
    runs=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for job in as_completed([pool.submit(evaluate,args) for args in tasks]):
            row=job.result();runs.append(row);print('eval',row['seed'],row['held_video'],flush=True)
    runs.sort(key=lambda r:(r['seed'],r['held_video']))
    values={name:np.zeros((2,80,3)) for name in ['zero','full','scalar','contextual','features']}
    for row in runs:
        for name in values:values[name][seeds.index(row['seed']),row['indices']]=row['costs'][name]
    summary=dict(zero=float(values['zero'].mean()))
    for name in ['full','scalar','contextual','features']:
        delta=values[name]-values['zero']
        summary[name]=dict(score=float(values[name].mean()),delta=float(delta.mean()),seed_deltas=delta.mean((1,2)).tolist(),
                           horizon_delta=delta.mean((0,1)).tolist(),per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])})
    summary['features_minus_scalar']=float((values['features']-values['scalar']).mean())
    summary['features_minus_motion']=float((values['features']-values['contextual']).mean())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,video=w['video'].tolist(),start=w['start'].tolist(),
                all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in runs),
                optimizer_failures=sum(not g['success'] for m in models for g in m['gates']),
                feature_selections=sum(g['use_features'] for m in models for g in m['gates']),
                note='All 5 controls frozen before new581017/18P32;allalpha2049/4097precision1e-4/1e-5. '
                     'Adapter/gate-levelCV only,backboneTRAIN;no hold/DEV/TEST/promotion. No feature/seed/regularization scan.')
    (root/'feature_soft_gate_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,gates=report['all_precision_gates_pass'],optimizer_failures=report['optimizer_failures'],feature_selections=report['feature_selections']),indent=2),flush=True)


if __name__=='__main__':main()
