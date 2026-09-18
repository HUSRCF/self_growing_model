"""Train on original prefixes plus early tails; evaluate purged late tails."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from validate_soft_gates_new_windows import causal_features
from validate_full_soft_gates import weights
from audit_crossfit_value import continuation
from crossfit_soft_kernel_gate import coefficients
from soft_output_mixture import value_gradient
from soft_kernel_gate import fit_gate as fit_motion, predict_gate as motion_predict
from feature_soft_kernel_gate import fit_mapping, design, fit_gate, predict_gate
from residual_soft_gate import block_starts, fit_gate as fit_residual, predict_gate as residual_predict
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def windows(block):
    out = {k: [] for k in ['history', 'truth', 'video', 'start']}
    for v in SPLITS['train'][:-3]:
        y = load_video(v)
        for t in block_starts(len(y), block):
            out['history'].append(y[t-31:t+1]); out['truth'].append(y[t+1:t+301])
            out['video'].append(v); out['start'].append(int(t))
    return {k: np.asarray(v) for k,v in out.items()}


def gather(kernel, motion, w, seed):
    p, f = continuation(AdaptiveBeam(), w['history'], seed, particles=32)
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(coefficients, kernel, motion[i:i+10], p[i:i+10,:,[49,99,299]],
                            w['truth'][i:i+10][:,[49,99,299]], f[i:i+10,:,[49,99,299]]) for i in range(0,len(motion),10)]
        rows = [j.result() for j in jobs]
    result = {size: {key: np.concatenate([r[0][size][key] for r in rows]).tolist()
                      for key in ['baseline','linear','quadratic']} for size in ['2049','4097']}
    return dict(coefficients=result, precision=[r[1] for r in rows], guards=int(f.any(-1).sum()))


def passes(row):
    return all(p['cost_pass'] and p['gradient_pass'] for p in row['precision'])


def new_weights(model, motion, raw):
    x = design(model['mapping'], raw)
    base = np.stack([motion_predict(g,np.log(motion+1e-12)) for g in model['motion']],1)
    return dict(new_scalar=np.broadcast_to([g['scalar'] for g in model['motion']],base.shape).copy(),
                new_motion=base, new_features=np.stack([predict_gate(g,x) for g in model['features']],1),
                residual=np.stack([residual_predict(g,x,base[:,t]) for t,g in enumerate(model['residual'])],1))


def main():
    root = Path('adaptive_search_results')
    paths = [root/'full_soft_gate_model.json',root/'full_soft_gate_training_terms.json',
             Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old = json.loads(paths[0].read_text())['model']; saved = json.loads(paths[1].read_text())
    assert saved['precision']['cost_pass'] and saved['precision']['gradient_pass']
    engine = AdaptiveBeam(); prefix = TrainingPrefixPool(engine.base,steps=300).sample(481017,per_video=8)
    np.testing.assert_array_equal(prefix['video'],saved['video']); np.testing.assert_array_equal(prefix['start'],saved['start'])
    early = windows('train')
    pm, pr = causal_features(engine,prefix); em, er = causal_features(engine,early)
    training = gather(old['kernel'],em,early,621017)
    training.update(video=early['video'].tolist(),start=early['start'].tolist())
    (root/'temporal_residual_training.json').write_text(json.dumps(training,indent=2))
    print('early-tail coefficients ready',training['precision'],flush=True)
    if not passes(training) or training['guards']:
        print('Training gate failure; no fitting or evaluation',flush=True); return
    terms = {k:np.concatenate([saved['coefficients']['4097'][k],training['coefficients']['4097'][k]])
             for k in ['baseline','linear','quadratic']}
    motion, raw = np.r_[pm,em],np.concatenate([pr,er]); mapping = fit_mapping(raw); x = design(mapping,raw)
    mg = [fit_motion(np.log(motion+1e-12),terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    fg = [fit_gate(x,terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    rg = [fit_residual(x,motion_predict(mg[t],np.log(motion+1e-12)),terms['linear'][:,t],terms['quadratic'][:,t]) for t in range(3)]
    model = dict(mapping=mapping,motion=mg,features=fg,residual=rg)
    replay = json.loads(json.dumps(model))
    for k,v in new_weights(model,motion,raw).items():
        np.testing.assert_array_equal(v,new_weights(replay,motion,raw)[k])
    model_path = root/'temporal_residual_model.json'
    model_path.write_text(json.dumps(dict(model=model,source_hashes=hashes,
        note='120 equally weighted windows:original80 prefix+40 earlytail. Fixed8projection;residual .25tanh with .001 allcoef L2, motion fallback. No evaluation loaded before freeze.'),indent=2))
    hashes[str(model_path)] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('models frozen', {k:[g['success'] for g in model[k]] for k in ['motion','features','residual']},flush=True)
    late = windows('evaluation')
    for v in SPLITS['train'][:-3]:
        assert early['start'][early['video']==v].max()+300 < late['start'][late['video']==v].min()-31
    lm, lr = causal_features(engine,late)
    alpha = {**{'old_'+k:v for k,v in weights(old,lm,lr).items()},**new_weights(model,lm,lr)}
    runs = []
    for seed in [622017,622018]:
        row = gather(old['kernel'],lm,late,seed)
        fine = {k:np.asarray(v) for k,v in row['coefficients']['4097'].items()}
        row.update(seed=seed,costs={k:value_gradient(fine,a)[0].tolist() for k,a in alpha.items()})
        runs.append(row); print('evaluation ready',seed,'precision',passes(row),flush=True)
    values = {k:np.asarray([r['costs'][k] for r in runs]) for k in alpha}
    summary = dict(zero=float(values['old_zero'].mean()))
    for k,val in values.items():
        if k=='old_zero':continue
        delta = val-values['old_zero']
        summary[k] = dict(score=float(val.mean()),delta=float(delta.mean()),seed_deltas=delta.mean((1,2)).tolist(),
                         horizon_delta=delta.mean((0,1)).tolist(),
                         per_video_delta={str(v):float(delta[:,late['video']==v].mean()) for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = dict(summary=summary,runs=runs,video=late['video'].tolist(),start=late['start'].tolist(),
                  alpha={k:v.tolist() for k,v in alpha.items()},source_hashes=hashes,
                  all_precision_gates_pass=all(passes(r) for r in runs),
                  note='New gate-training coverage protocol:fit10 only,earlytail training ends before3/4boundary,late evaluation histories start atboundary. '
                       'Old kernel/GRU/F frozen;prefix-trained kernel never fitted on tail labels. Historical videos/windows already inspected,NOT project-blind. '
                       'Two newRNG/P32,within-block overlap allowed,no tuning/hold/DEV/TEST/promotion.')
    (root/'temporal_residual_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
