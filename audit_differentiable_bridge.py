"""Real frozen-model forward parity and smooth fixed-path gradient checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,fixed_path
from feedback_distribution_pilot import sample
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS,ROOT
from v20_rnn_mixture.engine.data import continuous_features
from v20_rnn_mixture.engine.dynamics import mlp_features


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();bridge=FrozenBridge(s)
    files=[ROOT/'models/frozen_dynamics.json',ROOT/'models/gru_1901.npz'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    w=prefix_windows(SPLITS['train'][:-3],steps=10,per_video=4);h=w['history'];th=tensor(h);errors={}
    def check(name,actual,expected,tol=1e-10):
        actual=actual.detach().numpy() if isinstance(actual,torch.Tensor) else actual
        errors[name]=float(np.max(np.abs(actual-expected)));np.testing.assert_allclose(actual,expected,rtol=0,atol=tol)
    check('continuous_features',bridge.continuous_features(th),continuous_features(h,s.base))
    check('mlp_features',bridge.mlp_features(th),mlp_features(h,s.base.local['shared']['feature']))
    q,mem=s.machine.initialize(h);tq,tm=bridge.initialize(th)
    np.testing.assert_array_equal(tq,q);check('initial_hidden',tm,mem['hidden'])
    pe,T,read=s.machine.read(h,q,mem);a,b,c=bridge.read(th,tq,tm)
    check('event',a,pe);check('transition',b,T);check('read_hidden',c,read['read_hidden'])
    hh=np.repeat(h,64,axis=0);qq=np.tile(np.repeat(np.arange(8),8),len(h));rr=np.tile(np.tile(np.arange(8),8),len(h))
    check('all_64_edges',bridge.execute(tensor(hh),torch.tensor(qq),torch.tensor(rr)),s.base.execute_rule(hh,qq,rr))
    # Record normal NumPy paths; no tensor sampling or guards silently substituted.
    h=h[:4].copy();q,mem=s.machine.initialize(h);rng=np.random.default_rng(281017);events=[];destinations=[];pred=[];lp=np.zeros(len(h))
    for _ in range(10):
        pe,T,read=s.machine.read(h,q,mem);e=sample(pe,rng.random(len(h)));r=sample(T[np.arange(len(h)),e],rng.random(len(h)))
        lp+=np.log(pe[np.arange(len(h)),e])+np.log(T[np.arange(len(h)),e,r])
        y=s.base.execute_rule(h,q,r);assert np.isfinite(y).all() and (np.abs(y-h[:,-1])<=np.pi).all()
        events.append(e);destinations.append(r);pred.append(y);h=np.concatenate([h[:,1:],y[:,None]],1);q=r;mem={'hidden':read['read_hidden']}
    initial=tensor(w['history'][:4]);ee=torch.tensor(np.array(events));rr=torch.tensor(np.array(destinations))
    p,logs=fixed_path(bridge,initial,ee,rr,torch.zeros(2,dtype=torch.float64))
    check('ten_step_positions',p,np.stack(pred,1),tol=1e-9);check('ten_step_logprob',logs,lp,tol=1e-8)
    theta=torch.tensor([1e-5,-2e-5],dtype=torch.float64,requires_grad=True)
    truth=tensor(w['truth'][:4]);p,logs=fixed_path(bridge,initial,ee,rr,theta)
    cost=((torch.sin(p)-torch.sin(truth))**2+(torch.cos(p)-torch.cos(truth))**2).mean()
    vals={'path_loss':cost,'path_logprob':logs.mean()};gradient={}
    for name,value in vals.items():
        analytic=torch.autograd.grad(value,theta,retain_graph=True)[0].detach().numpy();finite=[]
        for dim in range(2):
            bump=torch.zeros(2,dtype=torch.float64);bump[dim]=1e-7;results=[]
            for sign in [1,-1]:
                pp,ll=fixed_path(bridge,initial,ee,rr,theta.detach()+sign*bump)
                results.append(float((((torch.sin(pp)-torch.sin(truth))**2+(torch.cos(pp)-torch.cos(truth))**2).mean() if name=='path_loss' else ll.mean()).detach()))
            finite.append((results[0]-results[1])/(2e-7))
        np.testing.assert_allclose(analytic,finite,rtol=1e-4,atol=1e-6)
        gradient[name]=dict(autograd=analytic.tolist(),finite_difference=finite)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(histories=len(th),max_abs_errors=errors,gradients=gradient,source_hashes=hashes,sources_unchanged=True,
        note='CPUfloat64 frozenbridge,TRAIN10prefix40histories/all64edges,10step4normalNumPy fixedpaths. Forward tolerances not bitwise equivalence. No training/samplingpolicy replacement/DEV/TEST. Fixed-path derivatives checked away from guards; hard failure behavior and long sampled trajectories not yet ported/audited.')
    Path('adaptive_search_results/differentiable_bridge_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
