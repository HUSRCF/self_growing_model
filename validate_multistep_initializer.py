"""Frozen n256 initializer models, eight new action RNG seeds on TRAIN hold."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from multistep_initializer_pilot import ResidualInitializer
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def paired_summary(differences):
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():raise ValueError('at least two finite paired differences required')
    return dict(differences=d.tolist(),mean=float(d.mean()),conditional_rng_se=float(d.std(ddof=1)/np.sqrt(len(d))),
                better=int((d<0).sum()),worse=int((d>0).sum()))


def main():
    root=Path('adaptive_search_results');paths=[root/'prefix_velocity_memory_model.npz']+[root/f'multistep_initializer_n256_h{h}.npz' for h in [1,10]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    s=AdaptiveBeam();_,Writer=legacy_types()
    weak=Writer(LegacyFeatureBridge(s.base),load_block(paths[0]),dict(learned_initialization=True))
    models=dict(original=None,base=weak)
    for h,path in zip([1,10],paths[1:]):
        with np.load(path) as z:models[f'h{h}']=ResidualInitializer(weak,z['coef'].copy(),z['scale'].copy())
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    for name,model in models.items():
        _,p,f,_=rollout_memory(s,w,model,201017)
        with np.load(root/f'multistep_initializer_n256_{name}_201017.npz') as old:
            for key,value in [('prediction',p),('failed',f),('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:
                np.testing.assert_array_equal(value,old[key])
    runs={name:[] for name in models}
    for seed in range(211017,211025):
        for name,model in models.items():
            r,p,f,_=rollout_memory(s,w,model,seed);runs[name].append(r)
            np.savez_compressed(root/f'validate_multistep_{name}_{seed}.npz',prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
            print(seed,name,r['objective'],flush=True)
    comparisons={}
    for candidate,ref in [('h1','base'),('h10','base'),('h1','original'),('h10','original'),('h10','h1')]:
        comparisons[f'{candidate}_minus_{ref}']=paired_summary([a['objective']-b['objective'] for a,b in zip(runs[candidate],runs[ref])])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(runs=runs,comparisons=comparisons,mean_objective={k:float(np.mean([r['objective'] for r in rs])) for k,rs in runs.items()},
        source_hashes=hashes,sources_unchanged=True,all_models_201017_exact=True,
        note='Frozen n256 h1/h10 plus original/base, eight new action seeds on same reused TRAIN hold24/P8/300. Conditional RNG SE only, not new-video uncertainty. No fit/tuning/selection/DEV/TEST/promotion.')
    (root/'multistep_initializer_validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
