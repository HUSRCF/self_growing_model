"""Post-hoc denser DEV diagnostic; not an independent holdout/test claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import run_policy
from v20_rnn_mixture.engine.data import tail_windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-seed',type=int,default=0)
    a=p.parse_args();root=Path('adaptive_search_results');s=AdaptiveBeam()
    arrays=None if a.model_seed==0 else dict(np.load(root/f'windows_uniform_seed{a.model_seed}_closed_loop.npz'))
    w=tail_windows('dev',300,32);runs=[]
    for seed in [1729,2718,3141]:
        result,pred,failed=run_policy(s,w,arrays=arrays,seed=seed,particles=8)
        runs.append(result)
        np.savez_compressed(root/f'refreshed128_model{a.model_seed}_roll{seed}.npz',prediction=pred,failed=failed,
                            truth=w['truth'],video=w['video'],window_start=w['start'])
        print(a.model_seed,seed,{t:result['score'][str(t)]['embedding_rmse'] for t in [50,100,300]},flush=True)
    report=dict(model_seed=a.model_seed,n_windows=128,runs=runs,
       note='Uniform-refresh no-feedback family chosen for this post-hoc denser DEV audit after32-window results. Checkpoints still TRAIN-selected. Same four DEV videos; not an independent confirmation or new test set. Start grids differ from32-window study.')
    (root/f'refreshed128_model{a.model_seed}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
