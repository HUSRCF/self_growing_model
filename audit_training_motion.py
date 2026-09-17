"""Read-only initial-history motion audit. No fitting, no TEST access."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.common import SPLITS, DT


def describe(s,windows):
    h=windows['history'];v=np.diff(h,axis=1)/DT
    motion=np.sqrt((v*v).mean((1,2)))
    q=s.base.state_from_history(h)[0]
    return dict(n_windows=len(h),velocity_rms_quantiles=np.quantile(motion,[0,.25,.5,.75,1]).tolist(),
                velocity_rms_mean=float(motion.mean()),q_counts=np.bincount(q,minlength=8).tolist(),
                per_video={str(v):float(motion[windows['video']==v].mean()) for v in np.unique(windows['video'])})


def main():
    s=AdaptiveBeam();train_tail=tail_windows('train',300,4)
    mask=np.isin(train_tail['video'],SPLITS['train'][:-3])
    train_tail={k:v[mask] for k,v in train_tail.items()}
    groups=dict(fit_prefix=prefix_windows(SPLITS['train'][:-3],per_video=4),
                selection_prefix=prefix_windows(SPLITS['train'][-3:],per_video=8),
                fit_videos_tail_diagnostic=train_tail,dev_tail=tail_windows('dev',300,8))
    result=dict(groups={k:describe(s,w) for k,w in groups.items()},
       note='RMS angular velocity from32 observed frames, rad/s; motion proxy, not physical energy. TRAIN tails used ONLY for this diagnostic, never fitting or checkpoint selection. Different time-position sampling; not a causal explanation of model error.')
    Path('adaptive_search_results/training_motion_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
