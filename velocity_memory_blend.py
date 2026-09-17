"""Fixed coarse blend using the legacy writer's synchronized velocity mapping.

The original-F branch maps displacement to velocity via 2*displacement-v.
This is not a trapezoidal identity for a nonlinear explicit-midpoint writer.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from velocity_memory_pilot import load_block,rollout_memory
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


class BlendedWriter:
    def __init__(self,base,weak,beta):
        if not 0<beta<=1:raise ValueError('beta0 must delegate original runtime')
        self.base=base;self.weak=weak;self.beta=beta
    def initialize(self,h):return self.weak.initialize(h)
    def execute(self,h,q,r,v):
        y,nv=self.weak.execute(h,q,r,v)
        if self.beta==1:return y,nv
        old=self.base.execute_rule(h,q,r)
        return (self.beta*y+(1-self.beta)*old,
                self.beta*nv+(1-self.beta)*(2*(old-h[:,-1])-v))


def main():
    root=Path('adaptive_search_results');path=root/'prefix_velocity_memory_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest();block=load_block(path)
    s=AdaptiveBeam();bridge=LegacyFeatureBridge(s.base);_,Writer=legacy_types()
    weak=Writer(bridge,block,dict(learned_initialization=True))
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs={}
    for beta in [0.,.25,.5,.75,1.]:
        name=str(beta);runs[name]=[];writer=None if beta==0 else BlendedWriter(s.base,weak,beta)
        for seed in [131017,131018,131019,131020]:
            result,p,f,_=rollout_memory(s,w,writer,seed);runs[name].append(result)
            if beta in [0,1]:
                oldname='original' if beta==0 else 'memory_learned'
                with np.load(root/f'prefix_velocity_memory_{oldname}_{seed}.npz') as old:
                    np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
            np.savez_compressed(root/f'velocity_memory_blend_{name}_{seed}.npz',prediction=p,failed=f,
                                truth=w['truth'],video=w['video'],window_start=w['start'])
        print(beta,np.mean([r['objective'] for r in runs[name]]),flush=True)
    means={name:float(np.mean([r['objective'] for r in values])) for name,values in runs.items()}
    selected=min(means,key=means.get);validation=[]
    if selected!='0.0':
        candidate=BlendedWriter(s.base,weak,float(selected))
        for seed in range(141017,141025):
            before,_,_,_=rollout_memory(s,w,None,seed)
            after,_,_,_=rollout_memory(s,w,candidate,seed)
            validation.append(dict(seed=seed,baseline=before,candidate=after,
                                   difference=after['objective']-before['objective']))
            print('validation',seed,validation[-1]['difference'],flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    delta=np.array([r['difference'] for r in validation])
    report=dict(runs=runs,mean_objective=means,selected_beta=float(selected),validation=validation,
                validation_mean_difference=float(delta.mean()) if len(delta) else None,
                conditional_rng_se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                model_sha256=digest,model_unchanged=True,both_endpoints_exact=True,
                note='Frozen writers; five predeclared betas; TRAIN hold selection. Eight fresh action seeds on the same videos only if a nonzero beta wins; otherwise validation is skipped. No fine sweep/refit/DEV/TEST. beta0 delegates original; beta1 identical weak. Intermediate velocity uses legacy synchronized mapping, not an exact nonlinear trapezoidal identity. Compound score primary; selection reuse is not independent-video proof.')
    (root/'velocity_memory_blend.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
