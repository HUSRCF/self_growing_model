"""Frozen correction counterfactuals at identical TRAIN states."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import correction
from feedback_distribution_pilot import sample
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def local_scores(centers,truth,probability,delta):
    """Exact categorical-law scores, not a finite sampled-ensemble estimate."""
    target=np.c_[np.sin(truth),np.cos(truth)]
    def values(y):
        emb=np.concatenate([np.sin(y),np.cos(y)],-1)
        mse=(probability*((emb-target[:,None])**2).mean(-1)).sum(1)
        distance=np.linalg.norm(emb-target[:,None],axis=-1)
        pair=np.linalg.norm(emb[:,:,None]-emb[:,None,:],axis=-1)
        energy=(probability*distance).sum(1)-.5*(probability[:,:,None]*probability[:,None,:]*pair).sum((1,2))
        return emb,mse,energy
    before,mse0,e0=values(centers);_,mse1,e1=values(centers+delta)
    tangent=np.concatenate([np.cos(centers)*delta,-np.sin(centers)*delta],-1)
    first_order=(probability*(2*(target[:,None]-before)*tangent).mean(-1)).sum(1)
    return dict(before_mse=mse0,after_mse=mse1,mse_gain=mse0-mse1,
                before_energy=e0,after_energy=e1,energy_gain=e0-e1,first_order_mse_gain=first_order,
                correction_rms=np.sqrt((probability[:,:,None]*delta**2).sum((1,2))/2))


def state_snapshots(s,history,seed,writer=None,truth=None,steps=100):
    h=history.copy();q,mem=s.machine.initialize(h);rng=np.random.default_rng(seed);dead=np.zeros(len(h),bool)
    result={}
    for t in range(steps+1):
        if t in [0,8,24,100]:result[t]=(h.copy(),q.copy(),mem['hidden'].copy(),dead.copy())
        if t==steps:break
        pe,tr,read=s.machine.read(h,q,mem)
        if truth is None:
            e=sample(pe,rng.random(len(h)));r=sample(tr[np.arange(len(h)),e],rng.random(len(h)))
            y=s.base.execute_rule(h,q,r)
            if writer is not None:y+=correction(writer,s.base,h,q,r)
            dead|=(~np.isfinite(y)).any(1)|(np.abs(y-h[:,-1])>np.pi).any(1);y[dead]=h[dead,-1]
        else:
            # Construction of observed diagnostic histories, not inference.
            y=truth[:,t]
        h=np.concatenate([h[:,1:],y[:,None]],1)
        q=s.base.state_from_history(h)[0] if truth is not None else r
        mem={'hidden':read['read_hidden']}
    return result


def main():
    root=Path('adaptive_search_results');source=json.loads((root/'continuous_residual_pilot.json').read_text())
    ridge=source['selected_ridge'];models={};hashes={}
    for name in ['constant',ridge]:
        path=root/source['model_files'][name];hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as a:models[name]={k:a[k].copy() for k in a.files}
        models[name]['kind']=str(models[name]['kind'].item())
    s=AdaptiveBeam();pieces=[]
    for video in SPLITS['train']:
        full=load_video(video);y=full[:len(full)//2];starts=np.linspace(63,len(y)-102,16,dtype=int)
        initial=np.stack([y[t-31:t+1] for t in starts]);truth=np.stack([y[t+1:t+102] for t in starts])
        for carrier in ['observed','zero','ridge']:
            for seed in ([0] if carrier=='observed' else [1729,2718,3141]):
                snapshots=state_snapshots(s,initial,seed+video,writer=models[ridge] if carrier=='ridge' else None,
                                          truth=truth if carrier=='observed' else None)
                for offset,(h,q,hidden,dead) in snapshots.items():
                    current_consistent=s.base.state_from_history(h)[0]==q
                    pe,tr,_=s.machine.read(h,q,{'hidden':hidden});p=np.einsum('be,ber->br',pe,tr)
                    hh=np.repeat(h,8,axis=0);qq=np.repeat(q,8);rr=np.tile(np.arange(8),len(h))
                    centers=s.base.execute_rule(hh,qq,rr).reshape(len(h),8,2)
                    diagnosed0=s.base.state_from_history(np.concatenate([hh[:,1:],centers.reshape(-1,1,2)],1))[0]
                    for name,model in models.items():
                        delta=correction(model,s.base,hh,qq,rr).reshape(len(h),8,2)
                        values=local_scores(centers,truth[:,offset],p,delta)
                        diagnosed=s.base.state_from_history(np.concatenate([hh[:,1:],(centers+delta).reshape(-1,1,2)],1))[0]
                        values.update(before_contradiction=(p*(diagnosed0.reshape(len(h),8)!=np.arange(8))).sum(1),
                                      after_contradiction=(p*(diagnosed.reshape(len(h),8)!=np.arange(8))).sum(1),
                                      diagnosis_flip_probability=(p*(diagnosed.reshape(len(h),8)!=diagnosed0.reshape(len(h),8))).sum(1),
                                      dead=dead,current_q_consistent=current_consistent,video=np.full(len(h),video),carrier=np.full(len(h),carrier),
                                      offset=np.full(len(h),offset),model=np.full(len(h),name),seed=np.full(len(h),seed))
                        pieces.append(values)
        print('TRAIN video',video,flush=True)
    rows={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};groups={}
    for split,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
        groups[split]={}
        for carrier in ['observed','zero','ridge']:
            groups[split][carrier]={}
            for name in models:
                dest={};groups[split][carrier][name]=dest
                for offset in [0,8,24,100]:
                    mask=np.isin(rows['video'],videos)&(rows['carrier']==carrier)&(rows['model']==name)&(rows['offset']==offset)
                    out={k:float(rows[k][mask].mean()) for k in ['before_mse','after_mse','mse_gain','before_energy','after_energy','energy_gain',
                         'first_order_mse_gain','correction_rms','before_contradiction','after_contradiction','diagnosis_flip_probability','dead']}
                    out.update(rows=int(mask.sum()),fraction_mse_helped=float((rows['mse_gain'][mask]>0).mean()),
                               relative_mse_gain=out['mse_gain']/max(out['before_mse'],1e-30),
                               per_video_mse_gain={str(v):float(rows['mse_gain'][mask&(rows['video']==v)].mean()) for v in videos})
                    out['current_q_groups']={}
                    for consistent in [False,True]:
                        subset=mask&(rows['current_q_consistent']==consistent)
                        out['current_q_groups'][str(consistent)]=dict(rows=int(subset.sum()),
                            mse_gain=float(rows['mse_gain'][subset].mean()) if subset.any() else None,
                            before_mse=float(rows['before_mse'][subset].mean()) if subset.any() else None)
                    dest[str(offset)]=out
    for name,digest in hashes.items():assert hashlib.sha256((root/source['model_files'][name]).read_bytes()).hexdigest()==digest
    np.savez_compressed(root/'residual_direction_rows.npz',**rows)
    report=dict(groups=groups,model_sha256=hashes,model_unchanged=True,
        note='TRAIN prefixes only; no parameter/strength fit, no oracle action selection, no DEV/TEST. Same-state counterfactual with original current event law. Carrier histories differ across observed/zero/ridge: within-carrier comparison is paired. Positive gain=helpful. Energy is exact local categorical mixture score, not sampled V/U estimate or full rollout score. Observed carrier uses real past; free carriers never read future truth. Repeated seeds/windows not independent videos; no significance claim.')
    (root/'residual_direction_audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
