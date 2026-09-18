"""Prespecified eight new RNG streams; frozen all candidates on both hold regions."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from confirm_temporal_kernel_holdout import hold_windows
from validate_soft_gates_new_windows import causal_features
from validate_group_risk_gate import predictions
from soft_kernel_gate import predict_gate
from audit_crossfit_value import continuation
from crossfit_soft_kernel_gate import coefficients
from soft_output_mixture import value_gradient


def evaluate(region,seed,model,kernel,prior):
    w=hold_windows(region);engine=AdaptiveBeam();motion,_=causal_features(engine,w)
    alpha=predictions(model['models'],model['reference'],motion)
    alpha.update(zero=np.zeros((len(motion),3)),
                 prior_motion=np.stack([predict_gate(g,np.log(motion+1e-12)) for g in prior],1))
    p,f=continuation(engine,w['history'],seed,particles=64)
    terms,precision=coefficients(kernel,motion,p[:,:,[49,99,299]],w['truth'][:,[49,99,299]],f[:,:,[49,99,299]])
    fine={k:np.asarray(v) for k,v in terms['4097'].items()}
    return dict(region=region,seed=seed,video=w['video'].tolist(),start=w['start'].tolist(),
                guards=int(f.any(-1).sum()),coefficients=terms,precision=precision,
                alpha={k:v.tolist() for k,v in alpha.items()},
                costs={k:value_gradient(fine,a)[0].tolist() for k,a in alpha.items()})


def main():
    root=Path('adaptive_search_results')
    paths=[root/'group_risk_gate_model.json',root/'temporal_kernel_model.json',root/'full_soft_gate_model.json',
           root/'temporal_kernel_training_audit.json',Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model=json.loads(paths[0].read_text());kernel_source=json.loads(paths[1].read_text())
    prior=json.loads(paths[2].read_text())['model']['motion_gates'];audit=json.loads(paths[3].read_text())
    assert kernel_source['all_converged'] and audit['all_precision_gates_pass']
    for parent in [model,audit]:
        for p,h in parent['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    runs=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs=[pool.submit(evaluate,region,seed,model,kernel_source['model'],prior)
              for region in ['prefix','tail'] for seed in range(641017,641025)]
        for job in as_completed(jobs):
            row=job.result();runs.append(row);print('done',len(runs),'of16',row['region'],row['seed'],row['precision'],flush=True)
    runs.sort(key=lambda r:(r['region'],r['seed']));summary={}
    for region in ['prefix','tail']:
        selected=[r for r in runs if r['region']==region];video=np.asarray(selected[0]['video'])
        for row in selected[1:]:
            for k,a in row['alpha'].items():np.testing.assert_array_equal(a,selected[0]['alpha'][k])
        values={k:np.array([r['costs'][k] for r in selected]) for k in selected[0]['costs']}
        out=dict(zero=float(values['zero'].mean()))
        for k,v in values.items():
            if k=='zero':continue
            delta=v-values['zero'];means=delta.mean((1,2))
            out[k]=dict(score=float(v.mean()),delta=float(delta.mean()),seed_deltas=means.tolist(),better_seeds=int((means<0).sum()),
                        conditional_seed_se=float(means.std(ddof=1)/np.sqrt(len(means))),
                        horizon_delta=delta.mean((0,1)).tolist(),
                        per_video_delta={str(v):float(delta[:,video==v].mean()) for v in np.unique(video)})
        summary[region]=out
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,
                all_precision_gates_pass=all(r['precision']['cost_pass'] and r['precision']['gradient_pass'] for r in runs),
                note='Prespecified641017-24/P64,all5risk candidates+zero/priorMotion,holdprefixANDtail24,16CPUworkers BLAS1. '
                     'No optional seed extension,refit,tuning,DEV/TEST/promotion. Fixed-window conditional MC check; '
                     'historically used hold/backboneTRAIN,NOT independent video generalization.')
    (root/'group_risk_confirmation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:dict(zero=v['zero'],worst_scalar=v['worst_scalar']) for k,v in summary.items()},indent=2),flush=True)


if __name__=='__main__':main()
