"""Frozen writers on prefix/tail regions of the same TRAIN-hold videos."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from confirm_truncated_writer import window_costs,paired
from validate_truncated_writer import load_models
from train_closed_loop_policy import prefix_windows
from training_window_sampler import TrainingPrefixPool
from v20_rnn_mixture.engine.common import SPLITS,DT
from v20_rnn_mixture.engine.data import tail_windows


def select_videos(w,videos):
    mask=np.isin(w['video'],videos)
    return {k:v[mask] for k,v in w.items()}


def motion(h):return np.sqrt(np.mean((np.diff(h,axis=1)/DT)**2,axis=(1,2)))


def state_summary(s,w,edges):
    m=motion(w['history']);q=s.base.state_from_history(w['history'])[0]
    return dict(motion_rms_quantiles=np.quantile(m,[0,.25,.5,.75,1]).tolist(),
                motion_bin_counts=np.bincount(np.searchsorted(edges,m,side='right'),minlength=4).tolist(),
                q_counts=np.bincount(q,minlength=8).tolist(),video=w['video'].tolist(),window_start=w['start'].tolist(),
                per_video={str(v):np.quantile(m[w['video']==v],[.25,.5,.75]).tolist() for v in np.unique(w['video'])})


def main():
    root=Path('adaptive_search_results');models,selections=load_models(root)
    source=root/'truncated_writer_confirmation.json';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    old=json.loads(source.read_text());assert selections==old['selections']
    s=AdaptiveBeam();prefix=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    tail=select_videos(tail_windows('train',300,8),SPLITS['train'][-3:])
    np.testing.assert_array_equal(tail['video'],prefix['video']);assert len(tail['history'])==24
    pool=TrainingPrefixPool(s.base,steps=300)
    edges=np.quantile(np.concatenate([p['motion'] for p in pool.pool.values()]),[.25,.5,.75])
    states={name:state_summary(s,w,edges) for name,w in [('prefix',prefix),('tail',tail)]}
    verified=set()
    for rows in old['runs'].values():
        for row in rows:
            if row['artifact'] in verified:continue
            with np.load(root/row['artifact']) as f:
                for k,key in [('truth','truth'),('video','video'),('window_start','start')]:np.testing.assert_array_equal(f[k],prefix[key])
                np.testing.assert_allclose(window_costs(f['prediction'],f['truth'],f['failed']),row['window_u_cost'],rtol=0,atol=1e-14)
            verified.add(row['artifact'])
    runs={name:[] for name in models}
    for seed in range(341017,341025):
        cache={}
        for name,model in models.items():
            key=(0.,0.) if model is None else tuple(model['value'])
            if key not in cache:
                row,p,f=rollout(s,tail,model,seed);c=window_costs(p,tail['truth'],f)
                np.testing.assert_allclose(c.mean(),row['objective'],rtol=0,atol=1e-14)
                row['window_u_cost']=c.tolist();row['per_video_u_cost']={str(v):float(c[tail['video']==v].mean()) for v in np.unique(tail['video'])}
                row['artifact']=f'writer_temporal_tail_{name}_{seed}.npz'
                np.savez_compressed(root/row['artifact'],prediction=p,failed=f,truth=tail['truth'],video=tail['video'],window_start=tail['start'])
                cache[key]=row
            runs[name].append(cache[key]);print(seed,name,cache[key]['objective'],flush=True)
    base=[r['objective'] for r in runs['zero']];summary={}
    for name,rows in runs.items():
        summary[name]=paired([r['objective'] for r in rows],base)
        summary[name]['prefix_delta']=old['summary'][name]['delta']
        summary[name]['per_video']={v:paired([r['per_video_u_cost'][v] for r in rows],[r['per_video_u_cost'][v] for r in runs['zero']]) for v in rows[0]['per_video_u_cost']}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    assert load_models(root)[1]==selections
    report=dict(selections=selections,runs=runs,summary=summary,states=states,fit_prefix_motion_edges=edges.tolist(),prefix_artifacts_verified=len(verified),prefix_source_sha=source_hash,
                note='Same three TRAIN-hold video identities,prefix vs tail,different windows,paired8action seeds341017–24. Descriptive temporal association,not controlled causal state intervention. Motion thresholds exclusively ten fit-video prefix pool,no labels/DEV. No fitting/reselection/DEV/TEST/default promotion.')
    (root/'writer_temporal_shift_audit.json').write_text(json.dumps(report,indent=2));print({k:{x:v[x] for x in ['objective','delta','prefix_delta']} for k,v in summary.items()},flush=True)


if __name__=='__main__':main()
