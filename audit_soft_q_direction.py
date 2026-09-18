"""Zero-strength soft Q directions on actual TRAIN states; no temperature fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from crossfit_conditional_influence import state_features
from evaluate_conditional_policy import predict_choices
from audit_soft_value import tilt


def directions(pe,t,q):
    pe,t,q=map(np.asarray,(pe,t,q))
    if (t.ndim!=3 or pe.shape!=t.shape[:2] or q.shape!=(len(t),t.shape[2])
        or not all(np.isfinite(a).all() for a in [pe,t,q]) or (pe<0).any() or (t<0).any()
        or not np.allclose(pe.sum(1),1) or not np.allclose(t.sum(2),1)):
        raise ValueError('Expected normalized pe[N,E],T[N,E,R],finite Q[N,R]')
    event_mean=np.einsum('ner,nr->ne',t,q)
    conditional=np.einsum('ne,ner->nr',pe,-t*(q[:,None]-event_mean[:,:,None]))
    prior=np.einsum('ne,ner->nr',pe,t);mean=np.sum(prior*q,1)
    joint=-prior*(q-mean[:,None])
    within=np.einsum('ne,ner->n',pe,t*(q[:,None]-event_mean[:,:,None])**2)
    between=np.sum(pe*(event_mean-mean[:,None])**2,1)
    for d in [conditional,joint]:np.testing.assert_allclose(d.sum(1),0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.sum(conditional*q,1),-within,atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.sum(joint*q,1),-within-between,atol=1e-12,rtol=0)
    return dict(conditional=conditional,joint=joint),within,between


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'conditional_state_coverage.json',directory/'temporal_q_coverage_model.json',directory/'q_prior_support_audit.json']
    cached,model,old=[json.loads(p.read_text()) for p in paths]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,h in model['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],cached[k])
    snap={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snap)
    xx=[];events=[];transitions=[];ys=[]
    for j,g in enumerate(cached['groups']):
        state={k:snap[50][k][j::8] for k in ['history','q','hidden','failed']}
        for k,h in g['state_hashes'].items():assert hashlib.sha256(state[k].tobytes()).hexdigest()==h
        x,prior=state_features(s,state)
        pe,t,_=s.machine.read(state['history'],state['q'],{'hidden':state['hidden']})
        np.testing.assert_array_equal(np.einsum('ne,ner->nr',pe,t),prior)
        np.testing.assert_array_equal(prior,g['prior'])
        xx.append(x);events.append(pe);transitions.append(t);ys.append(np.asarray([r['labels'] for r in g['rows']]))
    x,pe,t=np.concatenate(xx),np.concatenate(events),np.concatenate(transitions)
    y=np.concatenate(ys,axis=1);video=np.tile(w['video'],4);prior=np.einsum('ne,ner->nr',pe,t)
    choices,scores=predict_choices(model,x,video);results={};finite_errors={}
    for name,q in scores.items():
        np.testing.assert_array_equal(choices[name],old['results'][name]['choices'])
        np.testing.assert_array_equal(q[np.arange(len(q)),choices[name]]-(q*prior).sum(1),old['results'][name]['predicted_delta'])
        ds,within,between=directions(pe,t,q);finite_errors[name]={};results[name]={}
        for kind,d in ds.items():
            errors=[]
            for eps in [1e-3,5e-4]:
                if kind=='conditional':
                    plus=np.einsum('ne,ner->nr',pe,tilt(t,q,eps));minus=np.einsum('ne,ner->nr',pe,tilt(t,-q,eps))
                else:
                    plus=tilt(prior[:,None],q,eps)[:,0];minus=tilt(prior[:,None],-q,eps)[:,0]
                fd=(plus-minus)/(2*eps);np.testing.assert_allclose(fd,d,atol=1e-8,rtol=0)
                errors.append(float(np.max(abs(fd-d))))
            finite_errors[name][kind]=errors
            actual=np.einsum('nr,snrh->snh',d,y);pred=np.sum(d*q,1)
            results[name][kind]=dict(actual_slope=float(actual.mean()),seed_slope=actual.mean((1,2)).tolist(),
                horizon_slope=actual.mean((0,1)).tolist(),predicted_slope=float(pred.mean()),
                per_video_slope={str(v):float(actual[:,video==v].mean()) for v in np.unique(video)},
                group_slope=[float(actual[:,j*80:(j+1)*80].mean()) for j in range(4)],
                conditional_information=float(within.mean()),between_event_information=float(between.mean()),
                state_slopes=actual.tolist())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(results=results,finite_difference_errors=finite_errors,source_hashes=hashes,
                note='TRAIN320cached actualstates and811017-20P4labels, latest4LOVOcritics. Derivative atbeta0 ofeventconditional tilt preservingpe vs JOINT tilt ofpe*T by destinationQ (marginal destination tilt). Different policies; jointchanges eventmarginal. Frozen read_hidden beforeevent, future depends onr only. Normalization/support and predicted slope=-within variance (joint adds betweenevent variance) verified; centered FD1e-3/5e-4 atol1e-8,not performance temperature search. Influence derivative only,NOT finite fullscore gain. Prefixstate reconstruction only,no newfuturefit/rollouts/lateholdDEVTEST/defaultchange. Cachedresearchdata/sharedtruth/reference/streams,not freshvalidation or320independent states.')
    (directory/'soft_q_direction_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({n:{k:{a:v[a] for a in ['actual_slope','seed_slope','horizon_slope','predicted_slope','conditional_information','between_event_information']} for k,v in r.items()} for n,r in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
