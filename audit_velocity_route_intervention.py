"""Diagnostic reciprocal q-edge replay, not a deployable sampling policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from initial_velocity_distribution import InitialVelocityDistribution
from conditional_velocity_distribution import ConditionalVelocityDistribution
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from audit_conditional_trajectories import energy_parts
from evaluate_checked_particles import horizon_metrics
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def replay_edges(history,writer,sources,destinations,particles=8):
    sources=np.asarray(sources);destinations=np.asarray(destinations)
    h=np.repeat(history,particles,axis=0).copy()
    if sources.shape!=destinations.shape or sources.ndim!=2 or sources.shape[1]!=len(h):
        raise ValueError('edge arrays must be [time, window*particle]')
    q=sources[0].copy();v=writer.initialize(h);dead=~np.isfinite(v).all(1);v[dead]=0
    predictions=[];failures=[]
    for source,dest in zip(sources,destinations):
        np.testing.assert_array_equal(q[~dead],source[~dead])
        r=dest.copy()
        with np.errstate(over='ignore',invalid='ignore'):
            y,nv=writer.execute(h,q,r,v)
        dead|=(~np.isfinite(y)).any(1)|(~np.isfinite(nv)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        y[dead]=h[dead,-1];nv[dead]=v[dead];r[dead]=q[dead]
        nh=np.concatenate([h[:,1:],y[:,None]],1);nh[dead]=h[dead]
        predictions.append(y.copy());failures.append(dead.copy());h=nh;q=r;v=nv
    p=np.stack(predictions,1).reshape(len(history),particles,-1,2)
    return p,np.stack(failures,1).reshape(p.shape[:-1])


def main():
    root=Path('adaptive_search_results');model=root/'prefix_velocity_memory_model.npz'
    config=root/'conditional_velocity_distribution.json';globalfile=root/'initial_velocity_distribution.json'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,config,globalfile]}
    cfg=json.loads(config.read_text());cov=json.loads(globalfile.read_text())['covariance']
    s=AdaptiveBeam();_,Writer=legacy_types();weak=Writer(LegacyFeatureBridge(s.base),load_block(model),dict(learned_initialization=True))
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs=[]
    for seed in range(171017,171021):
        writers=dict(G=InitialVelocityDistribution(weak,cov,seed+1000000),
            C=ConditionalVelocityDistribution(weak,cfg['thresholds'],cfg['covariances'],seed+1000000))
        paths={};normal={}
        for name,writer in writers.items():
            _,p,f,trace=rollout_memory(s,w,writer,seed,trace=True);normal[name]=(p,f)
            paths[name]={key:np.stack([r[key] for r in trace]) for key in ['event','source','destination']}
            path=root/f'conditional_velocity_{"global" if name=="G" else "conditional"}_{seed}.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as old:
                np.testing.assert_array_equal(p,old['prediction']);np.testing.assert_array_equal(f,old['failed'])
            del trace
        cases={};costs={}
        for init,writer in writers.items():
            for route,path in paths.items():
                name=f'{init}_on_{route}'
                p,f=replay_edges(w['history'],writer,path['source'],path['destination'])
                if init==route:
                    np.testing.assert_array_equal(p,normal[init][0]);np.testing.assert_array_equal(f,normal[init][1])
                costs[name]=energy_parts(p,w['truth'],f)['total']
                cases[name]=dict(objective=float(costs[name][:,[49,99,299]].mean()),
                    score=horizon_metrics(p,w['truth'],f),
                    per_video_objective={str(v):float(costs[name][w['video']==v][:,[49,99,299]].mean()) for v in np.unique(w['video'])})
                np.savez_compressed(root/f'route_intervention_{name}_{seed}.npz',prediction=p,failed=f,
                    truth=w['truth'],video=w['video'],window_start=w['start'],**path)
        comparisons={}
        for name,a,b in [('closed_total','C_on_C','G_on_G'),('initial_on_G','C_on_G','G_on_G'),
                          ('routing_at_C','C_on_C','C_on_G'),('initial_on_C','C_on_C','G_on_C'),
                          ('routing_at_G','G_on_C','G_on_G')]:
            d=costs[a]-costs[b]
            comparisons[name]=dict(objective=float(d[:,[49,99,299]].mean()),
                horizons={str(t):float(d[:,t-1].mean()) for t in [50,100,300]},
                per_video={str(v):float(d[w['video']==v][:,[49,99,299]].mean()) for v in np.unique(w['video'])})
        np.testing.assert_allclose(comparisons['closed_total']['objective'],comparisons['initial_on_G']['objective']+comparisons['routing_at_C']['objective'],atol=1e-14)
        np.testing.assert_allclose(comparisons['closed_total']['objective'],comparisons['initial_on_C']['objective']+comparisons['routing_at_G']['objective'],atol=1e-14)
        mismatch={k:float((paths['G'][k]!=paths['C'][k]).mean()) for k in paths['G']}
        runs.append(dict(seed=seed,cases=cases,comparisons=comparisons,path_mismatch=mismatch))
        print(seed,{k:v['objective'] for k,v in comparisons.items()},flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(runs=runs,source_hashes=hashes,sources_unchanged=True,normal_and_self_replay_exact=True,
        note='Reciprocal fixed q-edge replay intervention, not normal conditional sampling. event metadata saved; numeric writer uses q/r only. GRU hidden unnecessary for exogenous edges, not used to choose actions. Failure freezes h/q/v. No truth-derived routing, no fit/tuning/DEV/TEST. Differences telescope but anchor-dependent interaction prevents unique causal attribution.')
    (root/'velocity_route_intervention.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
