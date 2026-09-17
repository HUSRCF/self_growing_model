"""TRAIN-only privileged one-step routing/center diagnostic; never a policy."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from feedback_distribution_pilot import sample
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def decomposition(centers,truth,probability):
    emb=np.concatenate([np.sin(centers),np.cos(centers)],-1)
    target=np.concatenate([np.sin(truth),np.cos(truth)],-1)
    if not np.isfinite(emb).all():raise ValueError('Nonfinite candidate needs separate handling')
    np.testing.assert_allclose(probability.sum(1),1,atol=1e-12)
    error=((emb-target[:,None])**2).mean(-1)
    mean=(probability[:,:,None]*emb).sum(1)
    expected=(probability*error).sum(1)
    mean_error=((mean-target)**2).mean(1)
    variance=(probability*((emb-mean[:,None])**2).mean(-1)).sum(1)
    oracle=error.argmin(1);ids=np.arange(len(truth))
    residual=np.angle(np.exp(1j*(truth-centers[ids,oracle])))
    return dict(expected_mse=expected,ensemble_mse=mean_error,candidate_variance=variance,
                oracle_mse=error[ids,oracle],routing_headroom=expected-error[ids,oracle],
                oracle_r=oracle,oracle_probability=probability[ids,oracle],
                modal_mse=error[ids,probability.argmax(1)],oracle_residual=residual)


def inspect(s,h,q,hidden,truth):
    # Read event law before enumerating its event/q/F candidate outcomes.
    pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
    probability=np.einsum('be,ber->br',pe,tr)
    centers=s.base.execute_rule(np.repeat(h,s.base.k,axis=0),np.repeat(q,s.base.k),np.tile(np.arange(s.base.k),len(h))).reshape(len(h),s.base.k,2)
    result=decomposition(centers,truth,probability)
    reject,_=s.checker.reject(np.repeat(h[:,:-1],s.base.k,axis=0),np.repeat(h[:,-1],s.base.k,axis=0),
                              centers.reshape(-1,2),np.repeat(q,s.base.k))
    valid=(~reject.reshape(len(h),s.base.k))&(np.abs(centers-h[:,-1,None]).max(2)<=20)
    emb=np.concatenate([np.sin(centers),np.cos(centers)],-1);target=np.concatenate([np.sin(truth),np.cos(truth)],-1)
    errors=((emb-target[:,None])**2).mean(-1)
    best=np.where(valid,errors,np.inf).min(1)
    next_history=np.concatenate([np.repeat(h[:,1:],s.base.k,axis=0),centers.reshape(-1,1,2)],axis=1)
    diagnosed=s.base.state_from_history(next_history)[0].reshape(len(h),s.base.k)
    consistent=diagnosed==np.arange(s.base.k)[None]
    consistent_best=np.where(consistent,errors,np.inf).min(1)
    result.update(current_check_support=valid.any(1),current_check_oracle_mse=np.where(np.isfinite(best),best,np.nan),
                  oracle_passes_current_check=valid[np.arange(len(h)),result['oracle_r']],q=q.copy(),
                  invalid_numeric_candidate_fraction=(np.abs(centers-h[:,-1,None]).max(2)>20).mean(1),
                  destination_consistent_support=consistent.any(1),
                  oracle_destination_consistent=consistent[np.arange(len(h)),result['oracle_r']],
                  consistent_oracle_mse=np.where(np.isfinite(consistent_best),consistent_best,np.nan))
    return result,pe,tr,read


def free_snapshots(s,history,seed,steps=100):
    h=history.copy();q,mem=s.machine.initialize(h);hidden=mem['hidden'];dead=np.zeros(len(h),bool)
    rng=np.random.default_rng(seed);snapshots={}
    for t in range(steps+1):
        if t in (0,24,100):snapshots[t]=(h.copy(),q.copy(),hidden.copy(),dead.copy())
        if t==steps:break
        pe,tr,read=s.machine.read(h,q,{'hidden':hidden})
        e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
        y=s.base.execute_rule(h,q,r)
        dead|=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1)
        y[dead]=h[dead,-1]
        h=np.concatenate([h[:,1:],y[:,None]],1);q=r;hidden=read['read_hidden']
    return snapshots


def summarize(rows,mask):
    out=dict(rows=int(mask.sum()),videos=sorted(np.unique(rows['video'][mask]).tolist()))
    for k in ['expected_mse','ensemble_mse','candidate_variance','oracle_mse','routing_headroom','modal_mse',
              'oracle_probability','current_check_support','oracle_passes_current_check','dead',
              'invalid_numeric_candidate_fraction','destination_consistent_support','oracle_destination_consistent']:
        out[k]=float(np.mean(rows[k][mask]))
    finite=mask&np.isfinite(rows['current_check_oracle_mse'])
    out['current_check_oracle_mse']=float(rows['current_check_oracle_mse'][finite].mean()) if finite.any() else None
    for name in ['current_check','consistent']:
        key=name+'_oracle_mse';paired=mask&np.isfinite(rows[key])
        out[key]=float(rows[key][paired].mean()) if paired.any() else None
        out[name+'_paired_expected_mse']=float(rows['expected_mse'][paired].mean()) if paired.any() else None
        out[name+'_paired_reduction_fraction']=1-out[key]/max(out[name+'_paired_expected_mse'],1e-30) if paired.any() else None
        out[name+'_paired_unrestricted_oracle_mse']=float(rows['oracle_mse'][paired].mean()) if paired.any() else None
        out[name+'_paired_oracle_gap']=float((rows[key][paired]-rows['oracle_mse'][paired]).mean()) if paired.any() else None
        if paired.any():assert (rows[key][paired]>=rows['oracle_mse'][paired]-1e-14).all()
    out['oracle_fraction_of_expected']=out['oracle_mse']/max(out['expected_mse'],1e-30)
    residual=rows['oracle_residual'][mask]
    out['oracle_absolute_residual_quantiles_rad']=np.quantile(np.linalg.norm(residual,axis=1),[.5,.9,.99]).tolist()
    out['per_video']={str(v):{k:float(rows[k][mask&(rows['video']==v)].mean()) for k in ['expected_mse','oracle_mse','ensemble_mse']}
                      for v in np.unique(rows['video'][mask])}
    return out


def main():
    s=AdaptiveBeam();pieces=[]
    def add(result,video,starts,offset,regime,seed,dead):
        result.update(video=np.full(len(starts),video),start=starts.copy(),offset=np.full(len(starts),offset),
                      regime=np.full(len(starts),regime),seed=np.full(len(starts),seed),dead=dead.copy())
        pieces.append(result)
    for video in SPLITS['train']:
        full=load_video(video);y=full[:len(full)//2]
        starts=np.linspace(63,len(y)-102,16,dtype=int)
        assert (starts+101<len(y)).all()
        initial=np.stack([y[t-31:t+1] for t in starts]);h=initial.copy()
        q,mem=s.machine.initialize(h);hidden=mem['hidden']
        for t in range(64):
            truth=y[starts+t+1];result,_,_,read=inspect(s,h,q,hidden,truth)
            add(result,video,starts,t,'observed',0,np.zeros(len(h),bool))
            h=np.concatenate([h[:,1:],truth[:,None]],1);q=s.base.state_from_history(h)[0];hidden=read['read_hidden']
        for seed in [1729,2718,3141]:
            for t,(h,q,hidden,dead) in free_snapshots(s,initial,seed+video).items():
                result,_,_,_=inspect(s,h,q,hidden,y[starts+t+1])
                add(result,video,starts,t,f'drift{t}',seed,dead)
        print('TRAIN video',video,'complete',flush=True)
    rows={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    np.testing.assert_allclose(rows['expected_mse'],rows['ensemble_mse']+rows['candidate_variance'],rtol=1e-12,atol=1e-14)
    assert (rows['routing_headroom']>=-1e-14).all()
    groups={}
    for split,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
        groups[split]={regime:summarize(rows,np.isin(rows['video'],videos)&(rows['regime']==regime))
                       for regime in ['observed','drift0','drift24','drift100']}
    root=Path('adaptive_search_results');np.savez_compressed(root/'writer_headroom_rows.npz',**rows)
    report=dict(groups=groups,variance_identity_verified=True,train_only=True,
        note='Only TRAIN prefixes; fit/hold denote prior adapter split, no fitting here. Observed chains16starts×64steps/video; free snapshots16starts×3RNG seeds/video at0/24/100. Drift0 repeats identical contexts. Oracle uses true next angle and is not deployable or a long-run bound. Oracle floor applies to selecting one of fixed candidate points, NOT their ensemble mean or a continuous extension. Ensemble here is local candidate mixture at one fixed history, not full rollout ensemble. Current-check support verifies the observed/generated middle, not a complete valid future. Destination consistency is a separate diagnostic restriction, not an official hard checker requirement. Restricted oracle comparisons use paired supported rows. Drift compares forecast history to aligned true trajectory, not observed one-step dynamics alone. No DEV/TEST or parameter updates.')
    (root/'writer_headroom.json').write_text(json.dumps(report,indent=2))
    for split,group in groups.items():
        for regime,r in group.items():print(split,regime,'expected',r['expected_mse'],'oracle',r['oracle_mse'],'consistent reduction',r['consistent_paired_reduction_fraction'])


if __name__=='__main__':main()
