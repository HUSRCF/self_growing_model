"""All gates frozen: different TRAIN-prefix starts and independent action streams."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_value_coverage import extra_windows
from train_search_value import features
from audit_crossfit_value import continuation
from crossfit_feature_soft_gate import evaluate
from feature_soft_kernel_gate import design,predict_gate


def causal_features(engine,w):
    motion=np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2)))
    q,mem=engine.machine.initialize(w['history']);_,_,read=engine.machine.read(w['history'],q,mem)
    raw=np.c_[features(engine.base,w['history'],q,read['read_hidden']),np.log(motion+1e-12)]
    return motion,raw


def prepare(args):
    seed,w,kernels,folds,models,raw,motion=args
    p,f=continuation(AdaptiveBeam(),w['history'],seed,particles=32)
    truth=w['truth'][:,[49,99,299]];tasks=[]
    for kernel,fold,model in zip(kernels,folds,models):
        assert kernel['held_video']==fold['held_video']==model['held_video']
        ids=np.flatnonzero(w['video']==kernel['held_video'])
        base=(seed,kernel,fold,ids,motion[ids],p[ids][:,:,[49,99,299]],truth[ids],f[ids][:,:,[49,99,299]])
        tasks.append((base,model,raw[ids]))
    return tasks


def main():
    root=Path('adaptive_search_results')
    paths=[root/'crossfit_output_kernel_models.json',root/'soft_kernel_gate_models.json',root/'feature_soft_gate_models.json',
           root/'feature_soft_gate_evaluation.json',Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    kernels=json.loads(paths[0].read_text())['models'];folds=json.loads(paths[1].read_text())['folds']
    models=json.loads(paths[2].read_text())['models'];prior=json.loads(paths[3].read_text())
    engine=AdaptiveBeam();pool=TrainingPrefixPool(engine.base,steps=300);old=pool.sample(481017,per_video=8)
    _,old_raw=causal_features(engine,old)
    for row in prior['runs']:
        model=next(m for m in models if m['held_video']==row['held_video'])
        x=design(model['mapping'],old_raw[row['indices']])
        alpha=np.stack([predict_gate(g,x) for g in model['gates']],1)
        np.testing.assert_array_equal(alpha,row['alpha']['features'])
    w=extra_windows(pool,old,count=8,seed=591017)
    assert not set(zip(w['video'],w['start'])) & set(zip(old['video'],old['start']))
    nearest=np.array([np.abs(old['start'][old['video']==v]-t).min() for v,t in zip(w['video'],w['start'])])
    motion,raw=causal_features(engine,w);seeds=list(range(592017,592021));tasks=[]
    print('old feature alpha exact; new80 starts frozen, overlapping old targets',int((nearest<300).sum()),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool_exec:
        for group in pool_exec.map(prepare,[(seed,w,kernels,folds,models,raw,motion) for seed in seeds]):tasks.extend(group)
    runs=[]
    with ProcessPoolExecutor(max_workers=8) as pool_exec:
        for job in as_completed([pool_exec.submit(evaluate,args) for args in tasks]):
            row=job.result();runs.append(row);print('done',len(runs),'of',len(tasks),flush=True)
    runs.sort(key=lambda r:(r['seed'],r['held_video']))
    values={name:np.zeros((4,80,3)) for name in ['zero','full','scalar','contextual','features']}
    for row in runs:
        for name in values:values[name][seeds.index(row['seed']),row['indices']]=row['costs'][name]
    summary=dict(zero=float(values['zero'].mean()))
    for name in ['full','scalar','contextual','features']:
        delta=values[name]-values['zero'];means=delta.mean((1,2))
        summary[name]=dict(score=float(values[name].mean()),delta=float(delta.mean()),seed_deltas=means.tolist(),
                           conditional_seed_se=float(means.std(ddof=1)/2),better_seeds=int((means<0).sum()),
                           horizon_delta=delta.mean((0,1)).tolist(),per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])})
    for control in ['scalar','contextual']:
        delta=values['features']-values[control]
        summary['features_minus_'+control]=dict(mean=float(delta.mean()),seed_deltas=delta.mean((1,2)).tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,video=w['video'].tolist(),start=w['start'].tolist(),
                old_feature_alpha_exact=True,old_nearest_start_distance=nearest.tolist(),
                overlapping_old_target_windows=int((nearest<300).sum()),overlapping_old_histories=int((nearest<32).sum()),
                all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in runs),
                note='Frozenall5 strategies,newTRAIN80prefix starts seed591017 excludes oldexact481017starts;same10videos. '
                     'New592017-20/P32. Different starts may overlap intervals,not independentvideos/globalblindtest. '
                     'Allfold kernels/gates/scales unchanged,allalpha2049/4097score1e-4/gradient1e-5. '
                     'No fitting/projection/regularization scan/hold/DEV/TEST/promotion;seedSE fixed-window only.')
    (root/'soft_gates_new_windows.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,gates=report['all_precision_gates_pass']),indent=2),flush=True)


if __name__=='__main__':main()
