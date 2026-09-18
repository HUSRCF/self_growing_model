"""Frozen two-width/nine-gate factorial comparison on both historical holdout regions."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_crossfit_value import continuation
from validate_soft_gates_new_windows import causal_features
from validate_full_soft_gates import weights
from validate_temporal_residual_gate import new_weights
from crossfit_soft_kernel_gate import coefficients
from soft_output_mixture import value_gradient
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def hold_windows(region):
    videos=SPLITS['train'][-3:]
    if region=='prefix':return prefix_windows(videos,steps=300,per_video=8)
    if region!='tail':raise ValueError('Unknown region')
    out={k:[] for k in ['history','truth','video','start']}
    for v in videos:
        y=load_video(v)
        for t in np.linspace(len(y)//2-1,len(y)-301,8,dtype=int):
            out['history'].append(y[t-31:t+1]);out['truth'].append(y[t+1:t+301])
            out['video'].append(v);out['start'].append(int(t))
    return {k:np.asarray(v) for k,v in out.items()}


def prepare(region,seed,old,new):
    w=hold_windows(region);engine=AdaptiveBeam()
    motion,raw=causal_features(engine,w)
    alpha={**{'old_'+k:v for k,v in weights(old,motion,raw).items()},**new_weights(new,motion,raw)}
    p,f=continuation(engine,w['history'],seed,particles=64)
    return dict(region=region,seed=seed,video=w['video'].tolist(),start=w['start'].tolist(),motion=motion,
                alpha=alpha,points=p[:,:,[49,99,299]],truth=w['truth'][:,[49,99,299]],
                failed=f[:,:,[49,99,299]],guards=int(f.any(-1).sum()))


def evaluate(row,name,kernel):
    terms,precision=coefficients(kernel,row['motion'],row['points'],row['truth'],row['failed'])
    fine={k:np.asarray(v) for k,v in terms['4097'].items()}
    return dict(region=row['region'],seed=row['seed'],kernel=name,coefficients=terms,precision=precision,
                costs={k:value_gradient(fine,a)[0].tolist() for k,a in row['alpha'].items()})


def main():
    root=Path('adaptive_search_results')
    paths=[root/'temporal_kernel_model.json',root/'temporal_kernel_training_audit.json',
           root/'temporal_residual_model.json',root/'full_soft_gate_model.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fitted=json.loads(paths[0].read_text());audit=json.loads(paths[1].read_text())
    assert fitted['all_converged'] and audit['all_precision_gates_pass']
    for p,h in audit['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    old=json.loads(paths[3].read_text())['model'];new=json.loads(paths[2].read_text())['model']
    kernels=dict(old=old['kernel'],new=fitted['model'])
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(prepare,region,seed,old,new) for region in ['prefix','tail'] for seed in range(631017,631021)]
        prepared=[j.result() for j in jobs]
    print('eight shared rollouts ready, all weights frozen',flush=True)
    for region in ['prefix','tail']:
        selected=[r for r in prepared if r['region']==region]
        for row in selected[1:]:
            for k,a in row['alpha'].items():np.testing.assert_array_equal(a,selected[0]['alpha'][k])
    runs=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs=[pool.submit(evaluate,row,name,kernel) for row in prepared for name,kernel in kernels.items()]
        for job in as_completed(jobs):
            row=job.result();runs.append(row);print('done',len(runs),'of16',row['region'],row['kernel'],row['seed'],row['precision'],flush=True)
    runs.sort(key=lambda r:(r['region'],r['kernel'],r['seed']));summary={}
    for region in ['prefix','tail']:
        metadata=next(r for r in prepared if r['region']==region);video=np.asarray(metadata['video'])
        selected={name:[r for r in runs if r['region']==region and r['kernel']==name] for name in kernels}
        baseline=np.asarray([r['costs']['old_zero'] for r in selected['old']])
        np.testing.assert_array_equal(baseline,[r['costs']['old_zero'] for r in selected['new']])
        result=dict(zero=float(baseline.mean()),strategies={})
        for gate in metadata['alpha']:
            if gate=='old_zero':continue
            old_values=np.asarray([r['costs'][gate] for r in selected['old']])
            for name in kernels:
                values=np.asarray([r['costs'][gate] for r in selected[name]])
                delta=values-baseline;seed_delta=delta.mean((1,2));change=values-old_values
                result['strategies'][name+'/'+gate]=dict(score=float(values.mean()),delta=float(delta.mean()),
                    width_change=float(change.mean()),seed_deltas=seed_delta.tolist(),seed_width_changes=change.mean((1,2)).tolist(),
                    conditional_seed_se=float(seed_delta.std(ddof=1)/2),better_seeds=int((seed_delta<0).sum()),
                    horizon_delta=delta.mean((0,1)).tolist(),
                    per_video_delta={str(v):float(delta[:,video==v].mean()) for v in np.unique(video)})
        summary[region]=result
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,
                metadata=[dict(region=r['region'],seed=r['seed'],video=r['video'],start=r['start'],guards=r['guards'],
                               alpha={k:v.tolist() for k,v in r['alpha'].items()}) for r in prepared],
                all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in runs),
                note='Frozen2kernels x9gates,hold18/16/13prefixANDtail24each,631017-20/P64,shared trajectories. '
                     '4rollout/16integralCPUworkers BLAS1. Historical hold/backboneTRAIN NOTblindtest. '
                     'Tail firsthistory may cross midpoint;within-region targets overlap. No fit/tuning/DEV/TEST/promotion.')
    (root/'temporal_kernel_holdout.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({r:dict(zero=s['zero'],new_old_motion=s['strategies']['new/old_contextual'],
                           new_motion=s['strategies']['new/new_motion']) for r,s in summary.items()},indent=2),flush=True)


if __name__=='__main__':main()
