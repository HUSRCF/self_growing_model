"""Observed-history one-step diagnostics with explicitly noncausal proxy controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_pilot import load_block,prefix_sequence
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def diagnostic_data(full,stride=8):
    y=prefix_sequence(full)
    t=np.arange(32,len(y)-4,stride)
    h=np.stack([y[i-31:i+1] for i in t])
    v=savgol_filter(y,9,4,deriv=1,axis=0,mode='interp')[t]
    a=savgol_filter(y,9,4,deriv=2,axis=0,mode='interp')[t]
    return h,y[t+1],v,a


def decompose(v,proxy_v,acc,proxy_a,current,truth):
    return np.stack([v-proxy_v,.5*(acc-proxy_a),current+proxy_v+.5*proxy_a-truth],axis=-2)


def main():
    root=Path('adaptive_search_results');path=root/'prefix_velocity_memory_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    s=AdaptiveBeam();_,Writer=legacy_types();from v19.writers import library
    writer=Writer(LegacyFeatureBridge(s.base),load_block(path),dict(learned_initialization=True))
    reports={}
    for split,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
        reports[split]={}
        for video in videos:
            h,truth,pv,pa=diagnostic_data(load_video(video));pieces=[]
            for start in range(0,len(h),128):
                hh=h[start:start+128];yy=truth[start:start+128];vp=pv[start:start+128];ap=pa[start:start+128]
                q,m=s.machine.initialize(hh);pe,T,_=s.machine.read(hh,q,m);prob=np.einsum('ne,ner->nr',pe,T)
                nh=np.concatenate([hh[:,1:],yy[:,None]],axis=1);true_r=s.base.state_from_history(nh)[0]
                n=len(hh);rr=np.tile(np.arange(s.base.k),n);qq=np.repeat(q,s.base.k)
                history=np.repeat(hh,s.base.k,axis=0);v=writer.initialize(history)
                proxy_v=np.repeat(vp,s.base.k,axis=0);proxy_a=np.repeat(ap,s.base.k,axis=0)
                target=np.repeat(yy,s.base.k,axis=0);current=history[:,-1]
                learned,_=writer.execute(history,qq,rr,v);proxy,_=writer.execute(history,qq,rr,proxy_v)
                coef=writer.coef[None]+writer.delta[qq,rr]
                acc=np.einsum('np,npd->nd',library(current,v,writer.d['order']),coef)
                terms=decompose(v,proxy_v,acc,proxy_a,current,target)
                np.testing.assert_allclose(terms.sum(1),learned-target,rtol=0,atol=1e-12)
                candidates=dict(original=s.base.execute_rule(history,qq,rr),learned=learned,proxy_velocity=proxy,
                                proxy_acceleration=current+v+.5*proxy_a,proxy_taylor=current+proxy_v+.5*proxy_a)
                vals={}
                weights=prob.reshape(-1)/n
                for name,pred in candidates.items():
                    error=np.angle(np.exp(1j*(pred-target))).reshape(n,s.base.k,2)
                    vals[name]=dict(weighted_mse=float(np.sum(prob*np.mean(error**2,axis=2))/n),
                        observed_edge_mse=float(np.mean(error[np.arange(n),true_r]**2)),
                        mean_error=np.sum(weights[:,None]*error.reshape(-1,2),axis=0).tolist())
                moments=np.einsum('n,nid,njd->ij',weights,terms,terms)/2
                pieces.append(dict(n=n,metrics=vals,moments=moments))
            total=sum(p['n'] for p in pieces)
            metrics={name:{key:np.average([p['metrics'][name][key] for p in pieces],axis=0,weights=[p['n'] for p in pieces]).tolist()
                           for key in pieces[0]['metrics'][name]} for name in pieces[0]['metrics']}
            moments=sum(p['n']*p['moments'] for p in pieces)/total
            reports[split][str(video)]=dict(n=total,metrics=metrics,residual_term_second_moments=moments.tolist(),raw_mse_from_terms=float(moments.sum()))
        print(split,'complete',flush=True)
    aggregate={}
    for split,rows in reports.items():
        rs=list(rows.values())
        aggregate[split]=dict(metrics={name:{key:np.mean([r['metrics'][name][key] for r in rs],axis=0).tolist()
            for key in rs[0]['metrics'][name]} for name in rs[0]['metrics']},
            residual_term_second_moments=np.mean([r['residual_term_second_moments'] for r in rs],axis=0).tolist())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(per_video=reports,aggregate=aggregate,model_sha256=digest,model_unchanged=True,
        note='TRAIN observed prefixes stride8; equal video aggregates, frozen probability-weighted all8destinations. Proxy v/a use future centered labels ONLY for offline diagnostics; observed-edge r also uses future truth. GRU initialized from each32frame observed history, not generated state. Residual terms velocity,half acceleration,Taylor remainder; full cross moments retained. No fit/rollout/DEV/TEST/promotion; proxy not measured physical truth.')
    (root/'writer_local_bias_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
