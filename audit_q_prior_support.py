"""Post-hoc TRAIN cache diagnostic of hard Q choices and original prior support."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from crossfit_conditional_influence import state_features
from evaluate_conditional_policy import predict_choices
from audit_delayed_state import correlation


def describe(p,delta,pred,video):
    # delta[streams,states], all descriptive: no threshold/gate fitting.
    y=delta.mean(0);logp=np.log(np.maximum(p,1e-12));bins=[]
    edges=[0.,.01,.1,.5,1.0000000001]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<hi);row=dict(lower=lo,upper=hi,n=int(mask.sum()))
        if mask.any():
            row.update(actual_delta=float(y[mask].mean()),positive_fraction=float(np.mean(y[mask]>0)),
                       conditional_stream_sd=float(delta[:,mask].std(0,ddof=1).mean()))
            if pred is not None:row.update(predicted_delta=float(pred[mask].mean()),optimism_gap=float((y[mask]-pred[mask]).mean()))
        bins.append(row)
    result=dict(n=len(p),selected_prior_quantiles=np.quantile(p,[0,.25,.5,.75,1]).tolist(),
                actual_delta=float(y.mean()),seed_delta=delta.mean(1).tolist(),positive_fraction=float(np.mean(y>0)),bins=bins,
                logprior_vs_delta=correlation(logp,y),logprior_vs_delta_within_video=correlation(logp,y,video),
                logprior_vs_stream_sd=correlation(logp,delta.std(0,ddof=1)))
    if pred is not None:
        result.update(predicted_delta=float(pred.mean()),optimism_gap=float((y-pred).mean()),
                      predicted_vs_actual_correlation=correlation(pred,y))
    return result


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'conditional_state_coverage.json',directory/'temporal_q_coverage_model.json']
    cached,model=[json.loads(p.read_text()) for p in paths]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,h in model['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],cached[k])
    snap={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snap)
    xs=[];priors=[];ys=[]
    for j,g in enumerate(cached['groups']):
        state={k:snap[50][k][j::8] for k in ['history','q','hidden','failed']}
        for k,h in g['state_hashes'].items():assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
        x,prior=state_features(s,state);np.testing.assert_array_equal(prior,g['prior'])
        xs.append(x);priors.append(prior);ys.append(np.asarray([r['labels'] for r in g['rows']]))
    x=np.concatenate(xs);prior=np.concatenate(priors);video=np.tile(w['video'],4)
    y=np.concatenate(ys,axis=1).mean(-1) # streams,states,actions; equal1s/3s.
    choices,scores=predict_choices(model,x,video)
    for kind in ['action','context']:
        np.testing.assert_array_equal(scores['prefix_'+kind],model['training_scores']['prefix_'+kind])
    choices['prior_mode']=prior.argmax(1)
    base=np.einsum('snr,nr->sn',y,prior);idx=np.arange(len(prior));results={}
    for name,choice in choices.items():
        p=prior[idx,choice];delta=y[:,idx,choice]-base
        pred=None if name not in scores else scores[name][idx,choice]-(scores[name]*prior).sum(1)
        group=[describe(p[j*80:(j+1)*80],delta[:,j*80:(j+1)*80],None if pred is None else pred[j*80:(j+1)*80],w['video']) for j in range(4)]
        results[name]=dict(aggregate=describe(p,delta,pred,video),groups=group,
                           selected_prior=p.tolist(),actual_delta_by_stream=delta.tolist(),
                           predicted_delta=None if pred is None else pred.tolist(),choices=choice.tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(results=results,source_hashes=hashes,video=video.tolist(),
                note='Post-hocTRAIN320actual t50states,stored811017-20/P4 actionlabels and811000reference. Latest prefix/mixed action/contextLOVOmodels, all labels for each predictedvideo excluded from its fold, but reused research data NOT fresh evaluation. No new future trajectories (only prefixstate reconstruction), no fit/thresholdsearch/late/hold/DEV/TEST/defaultchange. Fixed descriptive supportbins[0,.01,.1,.5,1];not actionable gates. Q influence only,not finite policy gains. Repeatedwindows/sharedtruth/reference/streams,not320independent samples. Existing event-conditional KL.01 softtilt and rootenergysoft negative/mixed evidence must be retained.')
    (directory/'q_prior_support_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['aggregate'] for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
