"""Frozen actual-state critics: finite distribution interventions at step51."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from crossfit_conditional_influence import state_features
from validate_temporal_root_critic import score
from resume_intervention import resume
from audit_energy_action_value import embedding
from audit_full_root_distribution import mixture_score,mixture_terms,change_terms


def predict_choices(model,x,video):
    result={};scores={}
    for name,folds in model['models'].items():
        values=np.empty(x.shape[:2])
        for v in np.unique(video):
            m=folds[str(v)]
            if m['contextual']:
                m=dict(m,arrays=dict(m['arrays'],projection=model['projection']))
            use=video==v;values[use]=score(m,x[use])
        scores[name]=values;result[name]=values.argmin(1)
    result['fixed_r6']=np.full(len(x),6,dtype=int)
    return result,scores


def evaluate(args):
    w,model,seed,particles=args;s=AdaptiveBeam();snap={50:None}
    p,f=continuation(s,w['history'],seed,particles=particles,snapshots=snap)
    state=snap[50];n=len(w['video']);x,_=state_features(s,state)
    choices,_=predict_choices(model,x,np.repeat(w['video'],particles))
    normal,nf=resume(s,state)
    np.testing.assert_array_equal(normal[:,0],p.reshape(-1,300,2)[:,50:])
    np.testing.assert_array_equal(nf[:,0],f.reshape(-1,300)[:,50:])
    baseline=None;mixed={};guards={'zero':int(f.any(-1).sum())};errors={};coefficients={}
    for name,choice in choices.items():
        suffix,sf=resume(s,state,root=choice)
        c=np.concatenate([p[:,:,:50],suffix.reshape(n,particles,250,2)],2)
        cf=np.concatenate([f[:,:,:50],sf.reshape(n,particles,250)],2)
        np.testing.assert_array_equal(c[:,:,:50],p[:,:,:50]);np.testing.assert_array_equal(cf[:,:,:50],f[:,:,:50])
        base=[];mix=[];linear=[];quadratic=[]
        for t in [49,99,299]:
            a,b=mixture_terms(embedding(np.stack([p[:,:,t],c[:,:,t]],1)),embedding(w['truth'][:,t]),np.stack([f[:,:,t],cf[:,:,t]],1))
            prior=np.broadcast_to([1.,0.],a.shape);prob=np.broadcast_to([.75,.25],a.shape)
            z,m=mixture_score(a,b,prior),mixture_score(a,b,prob)
            l,q=change_terms(a,b,np.broadcast_to([0.,1.],a.shape),prior)
            np.testing.assert_allclose(m-z,.25*l+.25**2*q,atol=1e-14,rtol=0)
            linear.append(l);quadratic.append(q)
            base.append(z);mix.append(m)
        base=np.stack(base,1);mix=np.stack(mix,1)
        if baseline is None:baseline=base
        else:np.testing.assert_array_equal(baseline,base)
        errors[name]=float(np.max(abs(mix[:,0]-base[:,0])));assert errors[name]<=1e-14
        mix[:,0]=base[:,0];mixed[name]=mix.tolist();guards[name]=int(cf.any(-1).sum())
        coefficients[name]=dict(linear=np.stack(linear,1).tolist(),quadratic=np.stack(quadratic,1).tolist())
    return dict(seed=seed,baseline=baseline.tolist(),mixed=mixed,guards=guards,short_roundoff=errors,
                choice_counts={k:np.bincount(v,minlength=8).tolist() for k,v in choices.items()},coefficients=coefficients)


def summarize(rows,video):
    result={};deltas={}
    for name in rows[0]['mixed']:
        d=np.asarray([np.asarray(r['mixed'][name])-r['baseline'] for r in rows]);assert (d[:,:,0]==0).all()
        deltas[name]=d;seed=d.mean((1,2))
        result[name]=dict(delta=float(d.mean()),seed_delta=seed.tolist(),horizon_delta=d.mean((0,1)).tolist(),
                          seed_horizon_delta=d.mean(1).tolist(),conditional_seed_se=float(seed.std(ddof=1)/np.sqrt(len(seed))),
                          per_video_horizon={str(v):d[:,video==v].mean((0,1)).tolist() for v in np.unique(video)})
        l=np.asarray([r['coefficients'][name]['linear'] for r in rows])
        q=np.asarray([r['coefficients'][name]['quadratic'] for r in rows])
        result[name].update(linear_contribution=float((.25*l).mean()),quadratic_contribution=float((.25**2*q).mean()),
                            seed_linear_contribution=(.25*l).mean((1,2)).tolist(),
                            horizon_linear_contribution=(.25*l).mean((0,1)).tolist(),
                            horizon_quadratic_contribution=(.25**2*q).mean((0,1)).tolist())
    diff=deltas['context']-deltas['action']
    return dict(baseline=float(np.mean([r['baseline'] for r in rows])),policies=result,
                context_minus_action=float(diff.mean()),context_minus_action_seed=diff.mean((1,2)).tolist())


def main():
    directory=Path('adaptive_search_results');path=directory/'conditional_influence_critic_model.json'
    report_path=directory/'conditional_policy_evaluation.json'
    previous=json.loads(report_path.read_text()) if report_path.exists() else None
    model=json.loads(path.read_text());hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    for p,h in model['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],model[k])
    snap={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snap)
    state={k:snap[50][k][::8] for k in ['history','q','hidden','failed']}
    x,_=state_features(s,state);choices,scores=predict_choices(model,x,w['video'])
    for name in scores:
        np.testing.assert_array_equal(scores[name],model['scores'][name]);np.testing.assert_array_equal(choices[name],model['choices'][name])
    # Replay a historical fixed-r6 finite mixture, not only derivative labels.
    source=directory/'delayed_root_training.json';hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    old=json.loads(source.read_text())['rows'][0];assert old['seed']==761017
    replay=evaluate((w,model,761017,8));expected=[];base=[]
    for a,b in zip(old['horizon_attraction'],old['horizon_pair_distance']):
        a,b=np.asarray(a),np.asarray(b);prob=np.zeros_like(a);prob[:,0]=.75;prob[:,7]=.25
        expected.append(mixture_score(a,b,prob));base.append(a[:,0]-.5*b[:,0,0])
    np.testing.assert_array_equal(replay['baseline'],np.stack(base,1))
    expected=np.stack(expected,1)
    np.testing.assert_allclose(replay['mixed']['fixed_r6'],expected,atol=1e-14,rtol=0)
    replay_error=float(np.max(abs(np.asarray(replay['mixed']['fixed_r6'])-expected)))
    print('Frozen choices and historical finite mixture replay passed',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(w,model,seed,16) for seed in range(821017,821021)]):
            rows.append(row);print('Finite policy',row['seed'],'guards',row['guards'],flush=True)
    summary=summarize(rows,w['video'])
    if previous is not None:
        for actual,prior in zip(rows,previous['rows']):
            assert actual['seed']==prior['seed']
            np.testing.assert_array_equal(actual['baseline'],prior['baseline'])
            for name in actual['mixed']:np.testing.assert_array_equal(actual['mixed'][name],prior['mixed'][name])
        print('Same-seed finite scores exactly replayed while adding L/Q decomposition',flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,rows=rows,source_hashes=hashes,historical_replay_error=replay_error,
                note='Frozen LOVO actual-state action/context critics,NO refit. All particles/new821017-20/P16 original-prefix states,one t50 forced destination chosen causally perparticle. Compare full Bernoulli continuation-distribution mixture .75original+.25policy vs original; NOT multiplying derivative by.25 or fixed quota particles. SharedRNG index excluded acrosscomponents. Full 50/100/300 energy U,first50 exact. Same80TRAINinitial windows, new recursive states beyond firstparticlecritic coverage; backboneTRAIN/criticvideoCV only. No timing/strength sweep,feedbackevery-step,late/hold/DEV/TEST/default promotion.')
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
