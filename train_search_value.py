"""Train a small terminal-cost regressor on TRAIN-video prefixes only.

Targets describe a fixed greedy continuation policy, not optimal return.
Generated candidate states are scored against the recorded future; at inference
only the generated state, q and GRU memory are inputs.
"""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video, continuous_features


def features(base, h, q, hidden):
    return np.concatenate([continuous_features(h, base), np.eye(base.k)[q], hidden], axis=1)


class ValueModel:
    def __init__(self, path):
        self.a = dict(np.load(path))

    def predict(self, base, node):
        a = self.a
        x = features(base, node.history, np.array([node.q]), node.hidden)
        z = np.clip((x-a['mean'])/a['scale'], -8, 8)
        phi = np.c_[np.ones(len(z)), z, np.tanh(z@a['projection'])]
        return float(np.clip(phi@a['coef'], 0, 2)[0])


def train():
    s = AdaptiveBeam()
    xs, targets, videos = [], [], []
    for video in SPLITS['train']:
        y = load_video(video)
        prefix = y[:len(y)//2]
        hs = np.stack([prefix[t-31:t+1] for t in np.linspace(63, len(prefix)-52, 16, dtype=int)])
        starts = np.linspace(63, len(prefix)-52, 16, dtype=int)
        q, mem = s.machine.initialize(hs)
        pe, trans, read = s.machine.read(hs, q, mem)
        pairs = [(i, int(e), int(r)) for i in range(len(hs))
                 for e in s._top(pe[i], 2) for r in s._top(trans[i,e], 2)]
        ids = np.array([p[0] for p in pairs]); rs = np.array([p[2] for p in pairs])
        h = hs[ids]; qq = q[ids]
        pred = s.base.execute_rule(h, qq, rs)
        h = np.concatenate([h[:,1:],pred[:,None]],1)
        hidden = read['read_hidden'][ids]; qq = rs
        x = features(s.base,h,qq,hidden)
        losses = []
        # Include root error and 10/25/50-step errors under the SAME fixed
        # no-check greedy continuation for every candidate.
        for step in range(1,51):
            if step in (1,10,25,50):
                truth = prefix[starts[ids]+step]
                embed = np.c_[np.sin(pred),np.cos(pred)]
                target = np.c_[np.sin(truth),np.cos(truth)]
                losses.append(np.mean((embed-target)**2,axis=1))
            if step == 50: break
            pe2,tr2,read2 = s.machine.read(h,qq,{'hidden':hidden})
            joint = pe2[:,:,None]*tr2
            chosen = joint.reshape(len(h),-1).argmax(1)
            rr = chosen % s.base.k
            pred = s.base.execute_rule(h,qq,rr)
            h = np.concatenate([h[:,1:],pred[:,None]],1)
            qq,hidden = rr,read2['read_hidden']
        xs.append(x); targets.append(np.mean(losses,axis=0)); videos.extend([video]*len(x))
    x=np.concatenate(xs); target=np.concatenate(targets); videos=np.asarray(videos)
    # Hold out whole training videos to audit generalization; dev/test are
    # never used to fit normalization, regression or hyperparameters.
    hold=np.isin(videos,SPLITS['train'][-3:]); fit=~hold
    mean=x[fit].mean(0); scale=np.maximum(x[fit].std(0),1e-5)
    z=np.clip((x-mean)/scale,-8,8)
    projection=np.random.default_rng(1901).normal(size=(x.shape[1],128))/np.sqrt(x.shape[1])
    phi=np.c_[np.ones(len(z)),z,np.tanh(z@projection)]
    reg=np.eye(phi.shape[1])*10; reg[0,0]=0
    coef=np.linalg.solve(phi[fit].T@phi[fit]+reg,phi[fit].T@target[fit])
    pred=np.clip(phi@coef,0,2)
    report={'training_videos':SPLITS['train'][:-3], 'validation_videos':SPLITS['train'][-3:],
            'n_samples':len(x), 'target':'mean embedding MSE at 1/10/25/50 under greedy continuation',
            'validation_mse':float(np.mean((pred[hold]-target[hold])**2)),
            'constant_validation_mse':float(np.mean((target[fit].mean()-target[hold])**2)),
            'validation_correlation':float(np.corrcoef(pred[hold],target[hold])[0,1])}
    group_truth=target[hold].reshape(-1,4)
    group_pred=pred[hold].reshape(-1,4)
    selected=group_truth[np.arange(len(group_truth)),group_pred.argmin(1)]
    report.update(validation_selected_cost=float(selected.mean()),
                  validation_mean_candidate_cost=float(group_truth.mean()),
                  validation_oracle_cost=float(group_truth.min(1).mean()),
                  validation_selected_regret=float((selected-group_truth.min(1)).mean()))
    out=Path('adaptive_search_results');out.mkdir(exist_ok=True)
    np.savez(out/'value_ridge.npz',mean=mean,scale=scale,projection=projection,coef=coef)
    (out/'value_training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': train()
