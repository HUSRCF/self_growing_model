"""Fold-only point/kernel gates over frozen conditional kernels; exact point U terms."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from audit_periodic_output_kernel import score_endpoints
from training_window_sampler import TrainingPrefixPool
from spectral_kernel_energy import blocked_value
from soft_output_mixture import exact_point_terms, value_gradient, interval_grid_difference
from soft_kernel_gate import fit_gate, predict_gate


def coefficients(kernel, motion, points, truth, failed):
    results = {}
    for size in [2049,4097]:
        terms = {key:np.zeros((len(points),3)) for key in ['baseline','linear','quadratic']}
        for j in range(len(points)):
            specs=kernel['conditional'][int(motion[j]>=kernel['threshold'])]
            for t,spec in enumerate(specs):
                theta=None if spec['use_zero'] else np.log(spec['kappa'])
                row=blocked_value(points[j,:,t],truth[j,t],failed[j,:,t],size,theta,mixture=True)
                exact=exact_point_terms(row)
                for key in terms:
                    terms[key][j,t]=exact[key]
                assert value_gradient(exact,0)[0]==row['baseline']
                np.testing.assert_allclose(value_gradient(exact,1)[0],row.get('raw_cost',row['baseline']),atol=1e-13,rtol=0)
        results[str(size)]=terms
    bounds=interval_grid_difference(results['2049'],results['4097'])
    gates=dict(cost=float(bounds['cost'].max()),gradient=float(bounds['gradient'].max()),
               cost_pass=bool(bounds['cost'].max()<=1e-4),gradient_pass=bool(bounds['gradient'].max()<=1e-5))
    return {size:{key:val.tolist() for key,val in terms.items()} for size,terms in results.items()},gates


def train_fold(args):
    kernel,indices,motion,points,truth,failed=args
    results,gates=coefficients(kernel,motion,points,truth,failed)
    row=dict(held_video=kernel['held_video'],fit_indices=indices.tolist(),coefficients=results,precision=gates)
    if gates['cost_pass'] and gates['gradient_pass']:
        l=np.array(results['4097']['linear']);q=np.array(results['4097']['quadratic'])
        row['gates']=[fit_gate(np.log(motion+1e-12),l[:,t],q[:,t]) for t in range(3)]
    return row


def evaluate_fold(args):
    seed,kernel,gate,indices,motion,points,truth,failed=args
    results,precision=coefficients(kernel,motion,points,truth,failed)
    n=len(points)
    alpha=dict(zero=np.zeros((n,3)),full=np.ones((n,3)),
               scalar=np.broadcast_to([g['scalar'] for g in gate['gates']],(n,3)).copy(),
               contextual=np.stack([predict_gate(g,np.log(motion+1e-12)) for g in gate['gates']],1))
    terms={key:np.array(val) for key,val in results['4097'].items()}
    costs={name:value_gradient(terms,a)[0].tolist() for name,a in alpha.items()}
    return dict(seed=seed,held_video=kernel['held_video'],indices=indices.tolist(),coefficients=results,
                precision=precision,costs=costs,alpha={k:v.tolist() for k,v in alpha.items()},guards=int(failed.sum()))


def main():
    root=Path('adaptive_search_results')
    files=[root/'crossfit_output_kernel_models.json',Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    kernels=json.loads(files[0].read_text())['models']
    engine=AdaptiveBeam();w=TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8)
    motion=np.sqrt(np.mean(np.diff(w['history'],axis=1)**2,axis=(1,2)))
    p,f=continuation(engine,w['history'],482017,particles=32)
    points,truth,failed=p[:,:,[49,99,299]],w['truth'][:,[49,99,299]],f[:,:,[49,99,299]]
    old=json.loads((root/'energy_kernel_refinement.json').read_text())
    np.testing.assert_array_equal(score_endpoints(points,truth,failed),old['results']['4097']['baseline'])
    if failed.any():raise RuntimeError('Training guard; no exclusion')
    tasks=[]
    for kernel in kernels:
        ids=np.flatnonzero(w['video']!=kernel['held_video'])
        assert set(w['video'][ids])==set(kernel['fit_videos'])
        tasks.append((kernel,ids,motion[ids],points[ids],truth[ids],failed[ids]))
    trained=[]
    with ProcessPoolExecutor(max_workers=10) as pool:
        for job in as_completed([pool.submit(train_fold,args) for args in tasks]):
            row=job.result();trained.append(row);print('train fold',row['held_video'],row['precision'],flush=True)
    trained.sort(key=lambda row:row['held_video'])
    fit_pass=all(row['precision']['cost_pass'] and row['precision']['gradient_pass'] for row in trained)
    frozen=dict(folds=trained,source_hashes=hashes,train_precision_pass=fit_pass,
                note='Frozen prior conditional9video kernels. Gate feature log(historyRMS+1e-12),scale9videosonly,clipz±3. '
                     'Perhorizon scalar exact bounded quadratic minimum;affineclip gate b0[-2,3],b1[-2,2],start[.5,0], '
                     'penalty .001*b1²,LBFGS200. Nonconverged or notbetterregularizedTRAIN loss -> exactscalarfallback. '
                     'Fullalpha2049/4097score1e-4/gradient1e-5 before gate fitting;exactpointterms. No heldvideo labels in gatefit.')
    model_file=root/'soft_kernel_gate_models.json';model_file.write_text(json.dumps(frozen,indent=2))
    if not fit_pass:
        print('TRAIN coefficient precision failure preserved; no evaluation',flush=True);return
    hashes[str(model_file)]=hashlib.sha256(model_file.read_bytes()).hexdigest()
    tasks=[];seeds=[571017,571018]
    for seed in seeds:
        p,f=continuation(engine,w['history'],seed,particles=32)
        for kernel,gate in zip(kernels,trained):
            assert kernel['held_video']==gate['held_video']
            ids=np.flatnonzero(w['video']==kernel['held_video'])
            tasks.append((seed,kernel,gate,ids,motion[ids],p[ids][:,:,[49,99,299]],truth[ids],f[ids][:,:,[49,99,299]]))
    runs=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for job in as_completed([pool.submit(evaluate_fold,args) for args in tasks]):
            row=job.result();runs.append(row);print('eval',row['seed'],row['held_video'],flush=True)
    runs.sort(key=lambda r:(r['seed'],r['held_video']))
    values={name:np.zeros((2,80,3)) for name in ['zero','full','scalar','contextual']}
    for row in runs:
        for name in values:values[name][seeds.index(row['seed']),row['indices']]=row['costs'][name]
    summary=dict(zero=float(values['zero'].mean()))
    for name in ['full','scalar','contextual']:
        delta=values[name]-values['zero']
        summary[name]=dict(score=float(values[name].mean()),delta=float(delta.mean()),seed_deltas=delta.mean((1,2)).tolist(),
                           horizon_delta=delta.mean((0,1)).tolist(),per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])})
    summary['contextual_minus_scalar']=float((values['contextual']-values['scalar']).mean())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,video=w['video'].tolist(),start=w['start'].tolist(),
                all_precision_gates_pass=all(row['precision']['cost_pass'] and row['precision']['gradient_pass'] for row in runs),
                optimizer_failures=sum(not g['success'] for row in trained for g in row['gates']),
                affine_selections=sum(g['use_affine'] for row in trained for g in row['gates']),
                note='Adapter/gate level videoCV only;backbone originallyTRAIN. Same80states/new571017/18P32 after allgates frozen. '
                     'Full alpha1 usesrawsmoothed not oldanchored;all controls rescored consistently. '
                     'No hold/DEV/TEST/default promotion;fitRNG reused from kernel fitting,not independent calibration states.')
    (root/'soft_kernel_gate_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,gates=report['all_precision_gates_pass'],optimizer_failures=report['optimizer_failures'],affine_selections=report['affine_selections']),indent=2),flush=True)


if __name__=='__main__':main()
