"""Pure diagnostic of actual intervention states, using TRAIN data only."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_full_root_distribution import mixture_terms


def correlation(x,y,video=None):
    x=np.asarray(x,float).copy();y=np.asarray(y,float).copy()
    if video is not None:
        for v in np.unique(video):
            use=video==v;x[use]-=x[use].mean();y[use]-=y[use].mean()
    return None if x.std()<1e-14 or y.std()<1e-14 else float(np.corrcoef(x,y)[0,1])


def inspect(args):
    w,seed=args;engine=AdaptiveBeam();snapshots={0:None,50:None}
    p,f=continuation(engine,w['history'],seed,particles=8,snapshots=snapshots)
    # Instrumentation must not perturb trajectories, failures, or random consumption.
    plain,pf=continuation(engine,w['history'],seed,particles=8)
    np.testing.assert_array_equal(p,plain);np.testing.assert_array_equal(f,pf)
    np.testing.assert_array_equal(snapshots[50]['history'].reshape(len(p),8,32,2),p[:,:,18:50])
    features={};states={}
    for t,state in snapshots.items():
        assert not state['failed'].any()
        before={k:state[k].copy() for k in ['history','q','hidden']}
        pe,tr,read=engine.machine.read(state['history'],state['q'],{'hidden':state['hidden']})
        for k,a in before.items():np.testing.assert_array_equal(a,state[k])
        probability=np.einsum('ne,ner->nr',pe,tr)
        motion=np.sqrt(np.mean(np.diff(state['history'],axis=1)**2,axis=(1,2)))
        values=dict(motion=motion,p6=probability[:,6],q6=(state['q']==6).astype(float),
                    entropy=-np.sum(probability*np.log(np.maximum(probability,1e-300)),1),
                    hidden_rms=np.sqrt(np.mean(state['hidden']**2,1)))
        features[str(t)]={k:v.reshape(len(p),8).mean(1).tolist() for k,v in values.items()}
        states[str(t)]=dict(q_histogram=np.bincount(state['q'],minlength=8).tolist(),
                            motion_median=float(np.median(motion)),
                            per_particle_q=state['q'].reshape(len(p),8).tolist(),
                            array_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in before.items()})
    # Match the source's independent-index reduction order, not energy_costs'
    # equivalent full-matrix sum (which differs by floating-point roundoff).
    terms=[mixture_terms(embedding(p[:,:,t])[:,None],embedding(w['truth'][:,t]),f[:,:,t][:,None]) for t in [49,99,299]]
    costs=np.stack([a[:,0]-.5*b[:,0,0] for a,b in terms],1)
    return dict(seed=seed,features=features,states=states,baseline=costs.tolist(),guards=int(f.any(-1).sum()))


def main():
    root=Path('adaptive_search_results');source=root/'delayed_root_training.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();old=json.loads(source.read_text())
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    w=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],old[k])
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(inspect,[(w,r['seed']) for r in old['rows']]))
    for actual,previous in zip(rows,old['rows']):
        a,b=np.asarray(previous['horizon_attraction']),np.asarray(previous['horizon_pair_distance'])
        baseline=(a[:,:,0]-.5*b[:,:,0,0]).T
        np.testing.assert_array_equal(actual['baseline'],baseline)
    delta=np.asarray([.25*np.asarray(r['linear'])+.25**2*np.asarray(r['quadratic']) for r in old['rows']])
    # r6 full-mixture long-horizon cost change, not a per-particle advantage.
    labels=delta[:,:,6,1:].mean(-1);association={}
    for t in ['0','50']:
        association[t]={}
        for feature in rows[0]['features'][t]:
            values=np.asarray([r['features'][t][feature] for r in rows]);same=[];cross=[];within=[]
            for i in range(4):
                other=np.delete(labels,i,axis=0).mean(0)
                same.append(correlation(values[i],labels[i]))
                cross.append(correlation(values[i],other))
                within.append(correlation(values[i],other,w['video']))
            association[t][feature]=dict(same_stream=same,other3_streams=cross,other3_within_video=within)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    report=dict(rows=rows,association=association,video=w['video'].tolist(),start=w['start'].tolist(),source_sha256=digest,
                note='TRAIN80/old761017-20P8 only. Snapshots before reads0/50 copied from actual recursion incl q/hidden/RNG/failed,NOT reinitialization. Instrumented vsplain entire trajectories bitwise; t50history equals generated points19..50; pure reads; oldbaseline exact. Window ensemble-mean causal state summaries vs fixedr6/.25 long-horizon full-mixture delta. Other3RNG labels mitigate same-stream coupling but share futuretruth; no causal/perparticle-policy inference, no independent-particle significance. No gate/model fit/threshold choice/late/hold/DEV/TEST.')
    (root/'delayed_state_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(association=association,q_histograms={t:[r['states'][t]['q_histogram'] for r in rows] for t in ['0','50']}),indent=2),flush=True)


if __name__=='__main__':main()
