"""Forward-mode dual propagation and temporal injection decomposition."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.autograd import forward_ad
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from training_window_sampler import TrainingPrefixPool
from audit_long_path_gradient import cost_and_adv
from hybrid_writer_gradient import energy_u


def propagate(bridge,initial,destinations,bound,direction,truth,adv,block=None,horizons=(50,100,300)):
    """Forward AD, not reverse-over-reverse JVP; no reverse tape needed."""
    def unpack(x):
        p,t=forward_ad.unpack_dual(x)
        return p.detach().clone(),torch.zeros_like(p) if t is None else t.detach().clone()
    with torch.no_grad(),forward_ad.dual_level():
        theta=forward_ad.make_dual(torch.zeros_like(bound),direction)
        h=initial;q,hidden=bridge.initialize(h);lp=torch.zeros(len(h),dtype=h.dtype)
        pred=[];history=[];i=torch.arange(len(h))
        for step,r in enumerate(destinations):
            pe,T,hidden=bridge.read(h,q,hidden)
            lp=lp+torch.log((pe[:,:,None]*T).sum(1)[i,r])
            correction=theta*bound if block is None or block[0]<=step<block[1] else torch.zeros_like(bound)
            y=bridge.execute(h,q,r)+correction
            pred.append(y)
            yp,yt=unpack(y);_,ht=unpack(hidden);_,lt=unpack(lp)
            history.append(dict(position=yp,position_tangent=yt,hidden_tangent=ht,logprob_tangent=lt))
            h=torch.cat([h[:,1:],y[:,None]],1);q=r
        p=torch.stack(pred,1);costs=[]
        for t in horizons:
            x=p[:,t-1];y=truth[t-1];emb=torch.cat([torch.sin(x),torch.cos(x)],-1)[None]
            costs.append(energy_u(emb,torch.cat([torch.sin(y),torch.cos(y)],-1))[0])
        cost=torch.stack(costs).mean();values=torch.stack([cost,cost+(adv*lp).sum()])
        primal,tangent=unpack(values)
    return primal,tangent,history


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s);root=Path('adaptive_search_results')
    w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1)
    previous=json.loads((root/'long_path_gradient_audit.json').read_text())
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    rows=[]
    for old in previous['rows']:
        seed=old['seed'];j=old['window_index'];initial=tensor(np.repeat(w['history'][j:j+1],4,axis=0));truth=tensor(w['truth'][j])
        with torch.no_grad():a=sampled_path(b,initial,300,torch.zeros(2,dtype=torch.float64),seed+j*100000,trace=True)
        assert not a['failed'].any();rr=torch.stack([x['q'] for x in a['trace']])
        adv=torch.stack([cost_and_adv(a['prediction'],truth,t)[1] for t in [50,100,300]]).mean(0)
        columns=[];histories=[]
        for direction in torch.eye(2,dtype=torch.float64):
            _,g,h=propagate(b,initial,rr,bound,direction,truth,adv)
            np.testing.assert_array_equal(torch.stack([x['position'] for x in h],1),a['prediction'])
            columns.append(g.numpy());histories.append(h)
        forward=np.stack(columns,1);reverse=np.array(old['analytic'])
        np.testing.assert_allclose(forward,reverse,rtol=1e-8,atol=1e-8)
        snapshots=[]
        for t in [10,50,100,150,200,250,300]:
            snapshots.append(dict(step=t,**{key:float(np.linalg.norm(np.stack([h[t-1][key].numpy() for h in histories],-1)))
                                            for key in ['position_tangent','hidden_tangent','logprob_tangent']}))
        blocks=[]
        for start in range(0,300,50):
            gradient=np.stack([propagate(b,initial,rr,bound,d,truth,adv,block=(start,start+50))[1].numpy() for d in torch.eye(2,dtype=torch.float64)],1)
            blocks.append(dict(start=start,stop=start+50,gradient=gradient.tolist()))
        total=np.array([x['gradient'] for x in blocks]).sum(0)
        np.testing.assert_allclose(total,forward,rtol=1e-8,atol=1e-8)
        row=dict(seed=seed,video=old['video'],start=old['start'],forward_gradient=forward.tolist(),reverse_gradient=reverse.tolist(),
                 max_absolute_disagreement=float(np.abs(forward-reverse).max()),snapshots=snapshots,blocks=blocks,
                 block_sum_max_error=float(np.abs(total-forward).max()))
        rows.append(row);print(seed,row['max_absolute_disagreement'],snapshots[-1],flush=True)
    report=dict(rows=rows,note='True forward-mode dual AD with reverse tape disabled; same scalar formulas, not independent model implementation. Fixed actions,theta0,old frozen advantages. Each 50-step injection block changes tangent only, all primal trajectories identical. Block derivatives sum by linearity; not separate optimized models or causal attribution of model defects. No training/DEV/TEST; hard-boundary gradient remains outside scope.')
    (root/'forward_tangent_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
