"""TRAIN fixed-state action-value signal, independent future RNG streams."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_delayed_state import correlation
from resume_intervention import resume, influence_cost


def worker(args):
    state,truth,reference,seed=args;s=AdaptiveBeam();labels=[];guards=[]
    for root in range(8):
        p,f=resume(s,state,root=root,seed=seed,branches=4)
        labels.append(np.stack([influence_cost(embedding(p[:,:,t]),truth[:,h],reference[:,:,h],f[:,:,t])
                                for h,t in enumerate([49,249])],-1))
        guards.append(int(f.any(-1).sum()))
    return dict(seed=seed,labels=np.stack(labels,1).tolist(),guards=guards)


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'delayed_state_audit.json',Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads(paths[0].read_text());s=AdaptiveBeam()
    w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    snapshots={50:None};p,f=continuation(s,w['history'],761017,particles=8,snapshots=snapshots)
    state=snapshots[50]
    for k,h in old['rows'][0]['states']['50']['array_sha256'].items():
        assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
    # Both normal and intervened suffixes must match full original simulations.
    for root in [None,6]:
        rp,rf=resume(s,state,root=root)
        target,failed=(p,f) if root is None else continuation(s,w['history'],761017,particles=8,root=root,forced_step=50)
        np.testing.assert_array_equal(rp[:,0],target.reshape(-1,300,2)[:,50:])
        np.testing.assert_array_equal(rf[:,0],failed.reshape(-1,300)[:,50:])
    # Predetermined first particle per window; no truth-based state selection.
    state={k:state[k][::8].copy() for k in ['history','q','hidden','failed']}
    assert not state['failed'].any()
    pe,tr,_=s.machine.read(state['history'],state['q'],{'hidden':state['hidden']})
    prior=np.einsum('ne,ner->nr',pe,tr)
    # Independent full-prefix bank represents original forecast distribution,
    # NOT the distribution conditioned on this one selected intermediate state.
    ref,reff=continuation(s,w['history'],801000,particles=16)
    reference=embedding(ref[:,:,[99,299]]);truth=embedding(w['truth'][:,[99,299]])
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(worker,[(state,truth,reference,seed) for seed in range(801017,801021)]):
            rows.append(row);print('Conditional labels',row['seed'],'guards',sum(row['guards']),flush=True)
    y=np.asarray([r['labels'] for r in rows]);fit=y[:2].mean(0);evaluation=y[2:]
    fitmean=fit.mean(-1);evalmean=evaluation.mean((0,3))
    idx=np.arange(len(prior));reference_cost=np.einsum('snrh,nr->snh',evaluation,prior)
    choices={'fit_truth_oracle':fitmean.argmin(1),'evaluation_truth_oracle':evalmean.argmin(1),
             'prior_mode':prior.argmax(1),'fixed_r6':np.full(len(prior),6)}
    summaries={}
    for name,choice in choices.items():
        delta=evaluation[:,idx,choice]-reference_cost
        summaries[name]=dict(mean_delta=float(delta.mean()),seed_horizon_delta=delta.mean(1).tolist(),
                             per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])},
                             choices=choice.tolist())
    report=dict(rows=rows,summary=summaries,prior=prior.tolist(),video=w['video'].tolist(),start=w['start'].tolist(),
                state_q=state['q'].tolist(),source_hashes=hashes,reference_guards=int(reff.any(-1).sum()),
                centered_label_correlation=correlation((fitmean-fitmean.mean(1,keepdims=True)).ravel(),(evalmean-evalmean.mean(1,keepdims=True)).ravel()),
                fit_eval_best_agreement=float(np.mean(fitmean.argmin(1)==evalmean.argmin(1))),
                state_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()},
                note='80 TRAIN windows, first particle of fixed761017/P8 pre-read50 state. Exact normal/forced6 suffix and oldstatehash replay. All8 forced one-time roots,4 branches each,801017/18 fit diagnostic labels,801019/20 independent future evaluation RNG; identical actual states. Independent original full-prefix801000/P16 fixed reference bank. Q=expected(distance-to-truth minus distance-to-reference plus2failure),mean1s/3s; energy first variation only,NOT finite-mixture score or measured deployed gain. Shared finite reference bank and truth cause shared errors; RNG replication is conditional,not generalization. Fit/eval truth oracles undeployable; no learned model/gate/threshold/late/hold/DEV/TEST/default changes.')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (directory/'conditional_influence_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['summary','centered_label_correlation','fit_eval_best_agreement','reference_guards']},indent=2),flush=True)


if __name__=='__main__':main()
