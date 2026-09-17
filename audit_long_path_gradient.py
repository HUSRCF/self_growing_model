"""Fixed discrete paths: long-horizon sensitivity and finite-difference audit."""
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from training_window_sampler import TrainingPrefixPool
from hybrid_writer_gradient import energy_u
from ensemble_score_objective import energy_costs


def replay(bridge,initial,events,destinations,theta):
    h=initial;q,hidden=bridge.initialize(h);lp=torch.zeros(len(h),dtype=h.dtype)
    pred=[];logs=[];guard=0;i=torch.arange(len(h))
    for e,r in zip(events,destinations):
        pe,T,hidden=bridge.read(h,q,hidden)
        lp=lp+torch.log((pe[:,:,None]*T).sum(1)[i,r])
        y=bridge.execute(h,q,r)+theta
        guard+=int(((~torch.isfinite(y)).any(1)|((y-h[:,-1]).abs()>np.pi).any(1)).sum())
        pred.append(y);logs.append(lp);h=torch.cat([h[:,1:],y[:,None]],1);q=r
    return torch.stack(pred,1),torch.stack(logs,1),guard


def cost_and_adv(pred,truth,t):
    x=pred[:,t-1];y=truth[t-1];emb=torch.cat([torch.sin(x),torch.cos(x)],-1)[None]
    target=torch.cat([torch.sin(y),torch.cos(y)],-1)
    cost=energy_u(emb,target)[0]
    _,base=energy_costs(emb.detach().numpy(),target.detach().numpy()[None],np.zeros((1,len(pred)),bool))
    return cost,cost.detach()-tensor(base[0])


def fixed_adv_objectives(pred,logs,truth,adv):
    cost=torch.stack([cost_and_adv(pred,truth,t)[0] for t in [50,100,300]]).mean()
    return torch.stack([cost,cost+(adv*logs[:,-1]).sum()])


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s);root=Path('adaptive_search_results')
    w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1)
    old=json.loads((root/'marginal_event_gradient_audit.json').read_text())
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    rows=[]
    for seed,j in [(321017,0),(321020,9),(321023,7)]:
        initial=tensor(np.repeat(w['history'][j:j+1],4,axis=0));truth=tensor(w['truth'][j])
        with torch.no_grad():a=sampled_path(b,initial,300,torch.zeros(2,dtype=torch.float64),seed+j*100000,trace=True)
        assert not a['failed'].any()
        ee=torch.stack([x['event'] for x in a['trace']]);rr=torch.stack([x['q'] for x in a['trace']])
        theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        pred,logs,guard=replay(b,initial,ee,rr,theta*bound)
        np.testing.assert_array_equal(pred.detach(),a['prediction']);np.testing.assert_array_equal(logs[:,-1].detach(),a['marginal_logp'])
        assert guard==0
        adv=torch.stack([cost_and_adv(pred,truth,t)[1] for t in [50,100,300]]).mean(0)
        values=fixed_adv_objectives(pred,logs,truth,adv)
        analytic=np.stack([torch.autograd.grad(v,theta,retain_graph=True)[0].detach().numpy() for v in values])
        reference=next(x for x in old['rows'] if x['seed']==seed)['windows'][j]['marginal']
        np.testing.assert_allclose(analytic[1],reference,rtol=1e-12,atol=1e-9)
        growth=[]
        for t in [10,50,100,150,200,250,300]:
            c,_=cost_and_adv(pred,truth,t)
            cg=torch.autograd.grad(c,theta,retain_graph=True)[0].detach().numpy()
            lg=torch.autograd.grad(logs[:,t-1].mean(),theta,retain_graph=True)[0].detach().numpy()
            growth.append(dict(step=t,path_gradient=cg.tolist(),mean_logprob_gradient=lg.tolist()))
        finite=[]
        for epsilon in [1e-3,1e-5,1e-7,1e-9]:
            columns=[];hits=0
            for dim in range(2):
                bump=torch.zeros(2,dtype=torch.float64);bump[dim]=epsilon;v=[]
                for sign in [1,-1]:
                    with torch.no_grad():
                        pp,ll,gg=replay(b,initial,ee,rr,sign*bump*bound)
                        v.append(fixed_adv_objectives(pp,ll,truth,adv).numpy());hits+=gg
                columns.append((v[0]-v[1])/(2*epsilon))
            fd=np.stack(columns,1);relative=np.linalg.norm(fd-analytic,axis=1)/np.maximum(np.linalg.norm(analytic,axis=1),1e-12)
            finite.append(dict(epsilon=epsilon,gradients=fd.tolist(),relative_error=relative.tolist(),guard_hits=hits))
        row=dict(seed=seed,window_index=j,video=int(w['video'][j]),start=int(w['start'][j]),analytic=analytic.tolist(),growth=growth,finite_difference=finite)
        rows.append(row);print(seed,j,analytic.tolist(),[(x['epsilon'],x['relative_error']) for x in finite],flush=True)
    report=dict(rows=rows,objective_order=['path_cost','frozen_advantage_marginal_surrogate'],
                note='Fixed sampled e/r,normalized theta units times old bound; frozen baseline advantages in finite differences. Diagnostic not derivative of resampling objective or global unbiasedness proof. Two previously identified large-gradient paths plus first-window control. No fitting/DEV/TEST; no deletion of difficult windows. Replay reports guard crossings without pretending failure-free deployment semantics there.')
    (root/'long_path_gradient_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
