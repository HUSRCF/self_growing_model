"""Paired label-policy audit; train prefixes only, common candidate states.

No fitting or test selection. Constrained labels use fixed-depth-2 replanning
including the actual committed-middle check. Failure adds an explicit penalty.
"""
import json
import time
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_search_ranker import step
from v20_rnn_mixture.engine.data import load_video


def embedding_mse(pred, truth):
    return float(np.mean((np.r_[np.sin(pred),np.cos(pred)]-
                          np.r_[np.sin(truth),np.cos(truth)])**2))


def main():
    started=time.perf_counter(); search=AdaptiveBeam(min_depth=2,max_depth=2)
    rows=[]; rng=np.random.default_rng(9217)
    for video in (12,20,7):
        full=load_video(video); data=full[:len(full)//2]
        for prefix in np.linspace(63,len(data)-60,4,dtype=int):
            initial=data[prefix-31:prefix+1][None]
            q0,mem=search.machine.initialize(initial)
            h=initial; q=q0; hidden=mem['hidden']
            for drift in range(9):
                if drift in (0,8):
                    pe,tr,read=search.machine.read(h,q,{'hidden':hidden})
                    for e in search._top(pe[0],2):
                        for r in search._top(tr[0,e],2):
                            y=search.base.execute_rule(h,q,np.array([r]))
                            admissible=True
                            if drift:
                                reject,_=search.checker.reject(h[:,:-1],h[:,-1],y,q)
                                admissible=not bool(reject[0])
                            ch=np.concatenate([h[:,1:],y[:,None]],1)
                            cq=np.array([r]); cm=read['read_hidden'].copy()
                            gh,gq,gm=ch.copy(),cq.copy(),cm.copy()
                            losses_g=[];losses_c=[];failed=False; fail_step=None
                            for t in range(1,51):
                                truth=data[prefix+drift+t]
                                if t in (10,25,50):
                                    losses_g.append(embedding_mse(gh[0,-1],truth))
                                    losses_c.append(embedding_mse(ch[0,-1],truth))
                                if t==50:break
                                gh,gq,gm=step(search,gh,gq,gm)
                                if not failed:
                                    node,_=search.search(ch,int(cq[0]),cm,rng,check_root=True)
                                    if node is None:
                                        failed=True;fail_step=t+1
                                    else:ch,cq,cm=node.history,np.array([node.q]),node.hidden
                            rows.append(dict(video=video,prefix=int(prefix),drift=drift,event=int(e),q=int(r),
                                             prior=float(pe[0,e]*tr[0,e,r]),admissible=admissible,
                                             greedy_cost=float(np.mean(losses_g)),
                                             constrained_cost=float(np.mean(losses_c))+2*int(failed),
                                             failed=failed,fail_step=fail_step))
                if drift<8:h,q,hidden=step(search,h,q,hidden,rng)
        print('completed video',video,flush=True)
    groups={}
    for row in rows:
        groups.setdefault((row['video'],row['prefix'],row['drift']),[]).append(row)
    inversions=0; comparable=0;changed=0;eligible=0;regrets=[]
    for group in groups.values():
        # Ignore inadmissible initial edges and duplicate destinations when
        # measuring trajectory ranking. Keep original event rows in artifact.
        unique={row['q']:row for row in group if row['admissible']}
        values=list(unique.values())
        if len(values)<2:continue
        g=np.array([r['greedy_cost'] for r in values]);c=np.array([r['constrained_cost'] for r in values])
        eligible+=1;changed+=int(g.argmin()!=c.argmin());regrets.append(float(c[g.argmin()]-c.min()))
        for i in range(len(values)):
            for j in range(i):
                if abs(g[i]-g[j])>1e-8 and abs(c[i]-c[j])>1e-8:
                    comparable+=1;inversions+=int((g[i]-g[j])*(c[i]-c[j])<0)
    summary=dict(videos=[12,20,7],n_rows=len(rows),n_groups=len(groups),eligible_groups=eligible,
                 best_destination_changed=changed,comparable_pairs=comparable,inverted_pairs=inversions,
                 inversion_fraction=inversions/comparable if comparable else None,
                 constrained_regret_of_greedy_oracle=float(np.mean(regrets)) if regrets else None,
                 failed_candidates=sum(r['failed'] for r in rows),
                 inadmissible_initial_edges=sum(not r['admissible'] for r in rows),
                 seconds=time.perf_counter()-started,failure_penalty=2,
                 note='Small train-prefix diagnostic, grouped correlated samples, not a model performance claim.')
    surviving_pairs=0; surviving_inversions=0
    for group in groups.values():
        values=list({r['q']:r for r in group if r['admissible'] and not r['failed']}.values())
        for i,a in enumerate(values):
            for b in values[:i]:
                dg=a['greedy_cost']-b['greedy_cost']; dc=a['constrained_cost']-b['constrained_cost']
                if abs(dg)>1e-8 and abs(dc)>1e-8:
                    surviving_pairs+=1; surviving_inversions+=int(dg*dc<0)
    summary.update(surviving_pairs=surviving_pairs,surviving_inversions=surviving_inversions)
    Path('adaptive_search_results/continuation_label_audit.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
