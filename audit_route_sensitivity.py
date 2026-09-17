"""Pure-read factorial interventions on paired history/q/hidden snapshots."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from initial_velocity_distribution import InitialVelocityDistribution
from conditional_velocity_distribution import ConditionalVelocityDistribution
from velocity_memory_pilot import load_block,rollout_memory
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def interval_overlap(p,q):
    pc=np.cumsum(p,axis=-1);qc=np.cumsum(q,axis=-1)
    pl=np.concatenate([np.zeros_like(pc[:,:1]),pc[:,:-1]],-1)
    ql=np.concatenate([np.zeros_like(qc[:,:1]),qc[:,:-1]],-1)
    return np.maximum(0,np.minimum(pc[:,:,None],qc[:,None,:])-np.maximum(pl[:,:,None],ql[:,None,:]))


def shared_uniform_mismatch(pe,T,qe,U):
    overlap=interval_overlap(pe,qe)
    event_same=np.trace(overlap,axis1=1,axis2=2)
    destination_same=np.zeros(len(pe))
    for e in range(pe.shape[1]):
        for f in range(qe.shape[1]):
            agree=np.trace(interval_overlap(T[:,e],U[:,f]),axis1=1,axis2=2)
            destination_same+=overlap[:,e,f]*agree
    return np.clip(1-event_same,0,1),np.clip(1-destination_same,0,1)


def describe(values,mask):
    if not mask.any():return dict(n=0)
    return dict(n=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in values.items()})


def main():
    root=Path('adaptive_search_results');paths=[root/'prefix_velocity_memory_model.npz',root/'conditional_velocity_distribution.json',root/'initial_velocity_distribution.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cfg=json.loads(paths[1].read_text());cov=json.loads(paths[2].read_text())['covariance']
    s=AdaptiveBeam();_,Writer=legacy_types();weak=Writer(LegacyFeatureBridge(s.base),load_block(paths[0]),dict(learned_initialization=True))
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);times=[0,1,5,10,25,50,100,200,299];rows=[];collected={t:[] for t in times}
    h0=np.repeat(w['history'],8,axis=0);q0,m0=s.machine.initialize(h0)
    for seed in range(171017,171021):
        writers=[InitialVelocityDistribution(weak,cov,seed+1000000),ConditionalVelocityDistribution(weak,cfg['thresholds'],cfg['covariances'],seed+1000000)]
        traces=[]
        for name,writer in zip(['global','conditional'],writers):
            _,p,f,trace=rollout_memory(s,w,writer,seed,trace=True)
            path=root/f'conditional_velocity_{name}_{seed}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                np.testing.assert_array_equal(p,z['prediction']);np.testing.assert_array_equal(f,z['failed'])
            assert not f.any(),'This snapshot audit requires live trajectories'
            traces.append(trace)
        for t in times:
            states=[(h0,q0,m0['hidden']) if t==0 else (tr[t-1]['history'],tr[t-1]['destination'],tr[t-1]['hidden']) for tr in traces]
            before=[[a.copy() for a in state] for state in states]
            reads={}
            for bits in itertools.product([0,1],repeat=3):
                h,q,m=[states[bits[i]][i] for i in range(3)]
                pe,T,_=s.machine.read(h,q,{'hidden':m})
                reads[''.join(map(str,bits))]=(pe,T,pe[:,:,None]*T)
            a,b=states;distance=np.sqrt(np.mean(np.angle(np.exp(1j*(a[0]-b[0])))**2,axis=(1,2)))
            values=dict(history_wrapped_rms=distance,q_mismatch=(a[1]!=b[1]).astype(float),hidden_rms=np.sqrt(np.mean((a[2]-b[2])**2,axis=1)))
            reference=reads['000'][2]
            for key,(_,_,joint) in reads.items():
                values[f'joint_tv_{key}']=.5*np.abs(joint-reference).sum((1,2))
                values[f'destination_tv_{key}']=.5*np.abs(joint.sum(1)-reference.sum(1)).sum(1)
            e,r=shared_uniform_mismatch(*reads['000'][:2],*reads['111'][:2])
            values.update(expected_event_mismatch=e,expected_destination_mismatch=r,
                realized_event_mismatch=(traces[0][t]['event']!=traces[1][t]['event']).astype(float),
                realized_destination_mismatch=(traces[0][t]['destination']!=traces[1][t]['destination']).astype(float))
            for state,old in zip(states,before):
                for x,y in zip(state,old):np.testing.assert_array_equal(x,y)
            collected[t].append(values)
            rows.append(dict(seed=seed,read_step=t,all=describe(values,np.ones(len(distance),bool)),
                             small_history=describe(values,distance<.001)))
        print('seed complete',seed,flush=True)
    aggregate={}
    for t,items in collected.items():
        values={k:np.concatenate([v[k] for v in items]) for k in items[0]}
        aggregate[str(t)]=dict(all=describe(values,np.ones(len(values['q_mismatch']),bool)),
            small_history=describe(values,values['history_wrapped_rms']<.001))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(rows=rows,aggregate=aggregate,source_hashes=hashes,sources_unchanged=True,normal_replay_exact=True,reads_pure=True,
        note='bits order history/q/pre-read hidden;0 global,1 conditional. TV relative000, not additive causal contributions. Hybrids can be off-manifold. Small history threshold .001rad RMS does not imply equal q/hidden. Exact shared two-uniform inverse-CDF mismatch for current distributions, not minimal coupling. No fit/tuning/DEV/TEST.')
    (root/'route_sensitivity_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
