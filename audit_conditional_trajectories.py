"""Read-only decomposition of saved conditional/global velocity trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def energy_parts(pred,truth,failed):
    x=np.concatenate([np.sin(pred),np.cos(pred)],axis=-1)
    y=np.concatenate([np.sin(truth),np.cos(truth)],axis=-1)
    p=x.shape[1]
    attraction=np.linalg.norm(x-y[:,None],axis=-1).mean(1)
    pair=np.zeros_like(attraction)
    for i in range(p):pair+=np.linalg.norm(x-x[:,i:i+1],axis=-1).sum(1)
    spread=pair/(2*p*(p-1));penalty=2*failed.mean(1)
    return dict(attraction=attraction,spread=spread,penalty=penalty,
                total=attraction-spread+penalty)


def group_summary(delta,mask):
    # delta arrays [action seed, window, time], all windows retain equal weight.
    n=int(mask.sum())
    if not n:return dict(n=0)
    ix=[49,99,299]
    byseed=delta['total'][:,mask][:,:,ix].mean((1,2))
    return dict(n=n,seed_objective_differences=byseed.tolist(),
        mean_difference=float(byseed.mean()),contribution=float(byseed.mean()*n/len(mask)),
        horizons={str(t):{k:float(v[:,mask,t-1].mean()) for k,v in delta.items()}
                  for t in [10,25,50,100,150,200,250,300]})


def main():
    root=Path('adaptive_search_results');source=root/'conditional_velocity_distribution.json'
    original=json.loads(source.read_text());hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    speed=np.linalg.norm(w['history'][:,-1]-w['history'][:,-2],axis=1)
    group=np.searchsorted(original['thresholds'],speed,side='right')
    parts={name:[] for name in ['global','conditional']}
    for name in parts:
        for i,seed in enumerate(range(171017,171021)):
            path=root/f'conditional_velocity_{name}_{seed}.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                for key,ref in [('truth',w['truth']),('video',w['video']),('window_start',w['start'])]:
                    np.testing.assert_array_equal(z[key],ref)
                values=energy_parts(z['prediction'],z['truth'],z['failed'])
            np.testing.assert_allclose(values['total'][:,[49,99,299]].mean(),original['runs'][name][i]['objective'],rtol=0,atol=1e-14)
            parts[name].append(values)
    delta={k:np.stack([a[k]-b[k] for a,b in zip(parts['conditional'],parts['global'])]) for k in parts['global'][0]}
    groups={'all':np.ones(len(speed),bool),**{f'video{v}':w['video']==v for v in np.unique(w['video'])},
            **{f'speed{g}':group==g for g in range(4)}}
    summary={k:group_summary(delta,mask) for k,mask in groups.items()}
    for prefix in ['video','speed']:
        np.testing.assert_allclose(sum(s.get('contribution',0) for k,s in summary.items() if k.startswith(prefix)),summary['all']['mean_difference'],rtol=0,atol=1e-14)
    windows=[]
    for i in range(len(speed)):
        mask=np.arange(len(speed))==i
        windows.append(dict(index=i,video=int(w['video'][i]),start=int(w['start'][i]),
            speed=float(speed[i]),speed_group=int(group[i]),**group_summary(delta,mask)))
    ranked=sorted(windows,key=lambda r:r['mean_difference'],reverse=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(groups=summary,windows=windows,descending_harm_window_indices=[r['index'] for r in ranked],
        curve_difference={k:v.mean((0,1)).tolist() for k,v in delta.items()},
        source_hashes=hashes,sources_unchanged=True,objectives_replayed=True,
        note='Saved trajectories only, no fit/new rollout/DEV/TEST. U-energy = attraction - spread + failure penalty; larger spread alone is not better calibration. Every window retained, ranking posthoc diagnostic only. Group contributions weighted by original window fraction, not independently sampled observations.')
    (root/'conditional_trajectory_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='horizons'} for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
