"""TRAIN-prefix-only causal state/motion sampling; no DEV-derived thresholds."""
import numpy as np
from v20_rnn_mixture.engine.common import SPLITS, DT
from v20_rnn_mixture.engine.data import load_video


class TrainingPrefixPool:
    def __init__(self,base,videos=None,steps=100):
        self.videos=list(SPLITS['train'][:-3] if videos is None else videos)
        if not set(self.videos).issubset(SPLITS['train'][:-3]):
            raise ValueError('Pool must exclude selection, DEV and TEST videos')
        self.steps=steps;self.pool={};self.seen=set();self.sampled_q=[];self.sampled_motion=[]
        for video in self.videos:
            full=load_video(video);y=full[:len(full)//2].copy()
            starts=np.arange(63,len(y)-steps)
            if len(starts)<4:raise ValueError('Insufficient prefix for four windows')
            h=np.stack([y[t-31:t+1] for t in starts])
            q=base.state_from_history(h)[0]
            motion=np.sqrt(np.mean((np.diff(h,axis=1)/DT)**2,axis=(1,2)))
            cells=[];thresholds={}
            # Each video's observed q gets its OWN TRAIN-only motion quartiles.
            for label in np.unique(q):
                ids=np.flatnonzero(q==label)
                edges=np.quantile(motion[ids],[.25,.5,.75])
                bins=np.searchsorted(edges,motion[ids],side='right')
                thresholds[str(int(label))]=edges.tolist()
                for b in range(4):
                    members=ids[bins==b]
                    if len(members):cells.append(members)
            self.pool[video]=dict(y=y,starts=starts,q=q,motion=motion,cells=cells,thresholds=thresholds)

    def sample(self,seed,mode='uniform',per_video=4):
        if mode not in ['uniform','stratified']:raise ValueError('Unknown window sampling mode')
        rng=np.random.default_rng(seed);out={k:[] for k in ['history','truth','video','start']}
        for video in self.videos:
            p=self.pool[video]
            if mode=='uniform':ids=rng.choice(len(p['starts']),per_video,replace=False)
            else:
                chosen=rng.choice(len(p['cells']),per_video,replace=len(p['cells'])<per_video)
                ids=np.array([rng.choice(p['cells'][c]) for c in chosen])
            for i in ids:
                t=int(p['starts'][i]);y=p['y']
                out['history'].append(y[t-31:t+1]);out['truth'].append(y[t+1:t+self.steps+1])
                out['video'].append(video);out['start'].append(t)
                self.seen.add((video,t));self.sampled_q.append(int(p['q'][i]));self.sampled_motion.append(float(p['motion'][i]))
        return {k:np.asarray(v) for k,v in out.items()}

    def audit(self):
        return dict(videos=self.videos,steps=self.steps,eligible_windows=sum(len(p['starts']) for p in self.pool.values()),
                    sampled_windows=len(self.sampled_q),distinct_sampled_windows=len(self.seen),
                    sampled_q_counts=np.bincount(self.sampled_q,minlength=8).tolist(),
                    sampled_velocity_rms_quantiles=np.quantile(self.sampled_motion,[0,.25,.5,.75,1]).tolist() if self.sampled_motion else None,
                    per_video={str(v):dict(prefix_stop=len(p['y']),min_start=int(p['starts'][0]),max_start=int(p['starts'][-1]),
                        pool_q_counts=np.bincount(p['q'],minlength=8).tolist(),nonempty_cells=len(p['cells']),
                        motion_quartiles_by_q=p['thresholds']) for v,p in self.pool.items()},
                    note='All strata use observed causal histories inside fit-TRAIN prefixes. No future error or DEV threshold. Stratification changes training-state weighting; no importance correction.')
