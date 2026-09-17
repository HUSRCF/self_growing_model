"""Original stochastic GRU+mixture on the shared 300-step DEV windows."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.runtime import CheckedMachine
from v20_rnn_mixture.engine.evaluate import metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--per-video',type=int,default=4)
    ap.add_argument('--particles',type=int,nargs='+',default=[1,8])
    ap.add_argument('--output',default='adaptive_search_results/common300_original.json')
    args=ap.parse_args()
    windows=tail_windows('dev',300,args.per_video); results=[]
    for particles in args.particles:
        for seed in (1729,2718,3141):
            model=CheckedMachine(); start=time.perf_counter()
            pred,info=model.rollout(windows['history'],300,particles,seed)
            scores,_=metrics(pred,windows['truth'],info['failed'])
            per_video={}
            for v in np.unique(windows['video']):
                mask=windows['video']==v
                per_video[str(v)],_=metrics(pred[mask],windows['truth'][mask],info['failed'][mask])
            results.append(dict(particles=particles,seed=seed,seconds=time.perf_counter()-start,score=scores,per_video=per_video))
            print(particles,seed,scores['300']['embedding_rmse'],flush=True)
    Path(args.output).write_text(json.dumps(dict(
        video=windows['video'].tolist(),window_start=windows['start'].tolist(),results=results,
        note='Original stochastic rollout may revise earlier points by rollback; search commits sequentially. Same initial windows, different selection semantics.'),indent=2))


if __name__=='__main__':main()
