"""Replay representative failed sparse windows and inspect full local support."""
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam,rollout
from evaluate_adapted_checked import checked_prefix,verify_checks
from v20_rnn_mixture.engine.data import tail_windows


def main():
    root=Path('adaptive_search_results');w=tail_windows('dev',300,8);report=[];probes=[]
    for model,seed,window in [(2718,1729,16),(3141,2718,13)]:
        s=AdaptiveBeam(min_depth=2,max_depth=2,temperature=1,search_interval=5,
              depth_invariant_temperature=True,shared_writer=True,boundary_repair=True,
              policy_adapter=str(root/f'windows_uniform_seed{model}_search_adapter.json'))
        search=s.search;snapshots=[]
        def capture(h,q,hidden,rng,**kwargs):
            node,info=search(h,q,hidden,rng,**kwargs)
            if node is None:
                info['snapshot']=len(snapshots)
                snapshots.append((h.copy(),q,hidden.copy(),dict(kwargs,banned_q=list(kwargs.get('banned_q',())))))
            return node,info
        s.search=capture
        raw,fail,diag=rollout(s,w['history'][window:window+1],301,particles=8,seed=seed+window,rollback_budget=300,rollback_window=2)
        pred,failed,_=checked_prefix(raw,fail,w['history'][window:window+1],300)
        expected=np.load(root/f'adapted_sparse_model{model}_roll{seed}.npz')
        np.testing.assert_array_equal(pred[0],expected['prediction'][window]);np.testing.assert_array_equal(failed[0],expected['failed'][window])
        for d in diag:
            if d['rollback'] or 'snapshot' not in d:continue
            h,q,hidden,kw=snapshots[d['snapshot']]
            roots=s._root_options(h,q,hidden,check_root=True,full=True)
            allowed=[n for n in roots if n.q not in kw['banned_q']]
            representatives={n.q:n for n in allowed}
            successors={str(r):sorted({n.q for n in s._expand(node,full=True)}) for r,node in representatives.items()}
            banned_successors={str(r):sorted({n.q for n in s._expand(node,full=True)})
                               for r,node in {n.q:n for n in roots if n.q in kw['banned_q']}.items()}
            report.append(dict(model_seed=model,rollout_seed=seed,window=window,particle=d['particle'],
                first_invalid_returned_step=d['step']-1,terminal_search_step=d['step'],source_q=q,
                banned_q=kw['banned_q'],boundary_repair_attempted=d['repair_attempted'],
                full_root_q=sorted({n.q for n in roots}),unbanned_root_q=sorted(representatives),
                full_next_q_by_root=successors,banned_root_full_next_q=banned_successors,diagnostic=d))
        for name,settings,revision in [('widen_empty',dict(widen_on_empty=True),2),
                                        ('full_root_before_backtrack',{},2),
                                        ('every_step_depth2',dict(search_interval=1),2),
                                        ('revision3',{},3)]:
            config=dict(min_depth=2,max_depth=2,temperature=1,search_interval=5,
                depth_invariant_temperature=True,shared_writer=True,boundary_repair=True,
                policy_adapter=str(root/f'windows_uniform_seed{model}_search_adapter.json'))
            config.update(settings);probe=AdaptiveBeam(**config);start=time.process_time()
            if name=='full_root_before_backtrack':
                ordinary=probe.search
                def retry_full(h,q,hidden,rng,**kwargs):
                    node,info=ordinary(h,q,hidden,rng,**kwargs)
                    if node is None and not kwargs.get('full_root',False):
                        opts=dict(kwargs,full_root=True,allow_lookahead=True)
                        return ordinary(h,q,hidden,rng,**opts)
                    return node,info
                probe.search=retry_full
            raw,fail,ds=rollout(probe,w['history'][window:window+1],301,particles=8,seed=seed+window,
                                rollback_budget=300,rollback_window=revision)
            pp,ff,bb=checked_prefix(raw,fail,w['history'][window:window+1],300)
            qs=np.full((1,8,301),-1,dtype=int)
            for d in ds:
                if 'root_q' in d and not d['rollback']:qs[0,d['particle'],d['step']]=d['root_q']
            verify_checks(probe.checker,w['history'][window:window+1],pp,bb,qs[:,:,:300],ff)
            item=dict(model=model,rollout_seed=seed,window=window,probe=name,failed_particles=int(ff[:,:,-1].sum()),
                       max_rollback_distance=max(d['rollback_distance'] for d in ds),
                       expansions=probe.audit['expansions'],cpu_seconds=time.process_time()-start)
            probes.append(item);print(item,flush=True)
    out=dict(cases=report,probes=probes,note='Exact replay of two representative failing windows, not all9 failed particles. Full two-edge support ignores ranking but retains F/checker; does not prove global feasibility or optimal recovery depth. Counterfactual probes are only on chosen failed windows, not full benchmark improvements.')
    (root/'adapted_deadend_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
