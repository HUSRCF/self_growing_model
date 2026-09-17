"""Frozen versus manifest-verified policy under checked and sparse runtimes.

All returned predictions have a generated right neighbor for central checks.
Sparse rollout produces H+1 points and shifts failure validity back by one;
its old H-only endpoint protocol is NOT a comparison baseline here.
"""
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam,rollout
from stateless_policy_adapter import StatelessPolicyAdapter
from v20_rnn_mixture.engine.runtime import CheckedMachine
from v20_rnn_mixture.engine.checker import MixtureChecker
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.evaluate import metrics


def checked_prefix(pred,failed,history,steps):
    if pred.shape[2]<steps+1 or failed.shape[2]<steps+1:raise ValueError('Extra right point required')
    out=pred[:,:,:steps].copy();bad=failed[:,:,1:steps+1].copy()
    valid=np.where(bad.any(2),bad.argmax(2),steps)
    boundary=np.empty(pred.shape[:2]+(2,))
    for i in range(len(out)):
        for p in range(out.shape[1]):
            v=int(valid[i,p]);boundary[i,p]=pred[i,p,v]
            if v<steps:out[i,p,v:]=history[i,-1] if v==0 else out[i,p,v-1]
    return out,bad,boundary


def verify_checks(checker,history,pred,boundary,states,failed):
    n,p,h,_=pred.shape
    left=np.concatenate([np.broadcast_to(history[:,None,-1:,:],(n,p,1,2)),pred[:,:,:-1]],axis=2)
    right=np.concatenate([pred[:,:,1:],boundary[:,:,None]],axis=2)
    valid=np.where(failed.any(2),failed.argmax(2),h)
    for i in range(n):
        for j in range(p):
            if valid[i,j]>0:right[i,j,valid[i,j]-1]=boundary[i,j]
    keep=~failed
    if not keep.any():return dict(checked_points=0,violations=0,maximum_excess=None)
    qs=states[keep]
    if (qs<0).any():raise AssertionError('Missing q for a claimed valid point')
    score=checker.score(left[keep][:,None,:],pred[keep],right[keep],qs)
    excess=score-checker.threshold[qs]
    violations=(~np.isfinite(score))|(excess>1e-10)
    if violations.any():raise AssertionError(f'{violations.sum()} returned points fail the unchanged checker')
    return dict(checked_points=int(keep.sum()),violations=int(violations.sum()),maximum_excess=float(excess.max()))


def sparse_window(task):
    i,history,seed,manifest,*extra=task;start=time.process_time()
    mode=extra[0] if extra else 'sparse'
    search=AdaptiveBeam(min_depth=2,max_depth=2,temperature=1,search_interval=1 if mode=='full' else 5,
             depth_invariant_temperature=True,shared_writer=True,boundary_repair=True,policy_adapter=manifest,
             commit_probe=mode in ('probe','probe_cache','probe_fast'),
             rule_cache_size=4096 if mode in ('probe_cache','probe_fast') else 0,fast_probe=mode=='probe_fast')
    raw,fail,diag=rollout(search,history[None],301,particles=8,seed=seed+i,rollback_budget=300,rollback_window=2)
    pred,failed,boundary=checked_prefix(raw,fail,history[None],300)
    states=np.full((1,8,301),-1,dtype=int)
    for d in diag:
        if 'root_q' in d and not d['rollback']:states[0,d['particle'],d['step']]=d['root_q']
    states=states[:,:,:300]
    audit=verify_checks(search.checker,history[None],pred,boundary,states,failed)
    stats=dict(search.audit,rollbacks=sum(d['rollback'] for d in diag),
               max_rollback_distance=max(d['rollback_distance'] for d in diag),
               boundary_repairs=sum(d['repair_success'] for d in diag),cpu_seconds=time.process_time()-start)
    return pred[0],failed[0],boundary[0],states[0],audit,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-seed',type=int,default=0)
    p.add_argument('--mode',choices=['checked','sparse','probe','probe_cache','probe_fast','full'],default='checked');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path('adaptive_search_results')
    manifest=None if args.model_seed==0 else str(root/f'windows_uniform_seed{args.model_seed}_search_adapter.json')
    w=tail_windows('dev',300,8);runs=[]
    for seed in [1729,2718,3141]:
        wall=time.perf_counter();cpu=time.process_time()
        if args.mode=='checked':
            model=CheckedMachine()
            if manifest:model.machine=StatelessPolicyAdapter(model.machine,manifest)
            pred,info=model.rollout(w['history'],300,particles=8,seed=seed)
            failed=info['failed'];states=info['next_state']
            boundary=np.asarray(model.last_stats['last_unreturned_point']).reshape(len(pred),8,2)
            verification=verify_checks(model.checker,w['history'],pred,boundary,states,failed)
            stats={k:v for k,v in model.last_stats.items() if not isinstance(v,list)}
            stats['cpu_seconds']=time.process_time()-cpu
        else:
            tasks=[(i,h,seed,manifest,args.mode) for i,h in enumerate(w['history'])]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:items=list(pool.map(sparse_window,tasks))
            pred=np.array([r[0] for r in items]);failed=np.array([r[1] for r in items])
            boundary=np.array([r[2] for r in items]);states=np.array([r[3] for r in items])
            verification=verify_checks(MixtureChecker(),w['history'],pred,boundary,states,failed)
            stats={k:(max(r[5][k] for r in items) if k=='max_rollback_distance' else sum(r[5][k] for r in items)) for k in items[0][5]}
        score,_=metrics(pred,w['truth'],failed)
        runs.append(dict(seed=seed,score=score,verification=verification,stats=stats,seconds=time.perf_counter()-wall,
            per_video={str(v):metrics(pred[w['video']==v],w['truth'][w['video']==v],failed[w['video']==v])[0] for v in np.unique(w['video'])}))
        stem=f'adapted_{args.mode}_model{args.model_seed}'
        np.savez_compressed(root/f'{stem}_roll{seed}.npz',prediction=pred,failed=failed,states=states,boundary=boundary,
                            truth=w['truth'],video=w['video'],window_start=w['start'])
        report=dict(config=vars(args),manifest=manifest,runs=runs,
            note='Exploratory DEV32. Same policy family TRAIN-selected; no best optimizer seed. Every valid returned point centrally checked, including endpoint. Sparse uses301 internal outputs and validity shift; prior300-only sparse results not comparable. Checked and sparse have different rollback budgets and sampling semantics; compare adapters within runtime.')
        (root/(stem+'.json')).write_text(json.dumps(report,indent=2))
        print(args.mode,args.model_seed,seed,[round(score[str(t)]['embedding_rmse'],6) for t in [50,100,300]],verification,flush=True)


if __name__=='__main__':main()
