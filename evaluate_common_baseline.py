"""Original stochastic GRU+mixture on the shared 300-step DEV windows."""
import json
import time
from pathlib import Path
import numpy as np
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.runtime import CheckedMachine
from v20_rnn_mixture.engine.evaluate import metrics


def main():
    windows=tail_windows('dev',300,4); results=[]
    for particles in (1,8):
        for seed in (1729,2718,3141):
            model=CheckedMachine(); start=time.perf_counter()
            pred,info=model.rollout(windows['history'],300,particles,seed)
            scores,_=metrics(pred,windows['truth'],info['failed'])
            results.append(dict(particles=particles,seed=seed,seconds=time.perf_counter()-start,score=scores))
            print(particles,seed,scores['300']['embedding_rmse'],flush=True)
    Path('adaptive_search_results/common300_original.json').write_text(json.dumps(dict(
        video=windows['video'].tolist(),window_start=windows['start'].tolist(),results=results,
        note='Original stochastic rollout may revise earlier points by rollback; search commits sequentially. Same initial windows, different selection semantics.'),indent=2))


if __name__=='__main__':main()
