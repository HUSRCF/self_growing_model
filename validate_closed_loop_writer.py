"""Fixed all-seed writer candidates, fresh holdout action RNG comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');models={'zero':None};sources={};hashes={}
    for seed in [1901,2718,3141]:
        path=root/f'closed_loop_writer_seed{seed}.npz';hashes[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:models[str(seed)]={k:z[k].copy() for k in z.files}
        sources[str(seed)]=json.loads((root/f'closed_loop_writer_seed{seed}.json').read_text())
    baseline=[r['objective'] for r in sources['1901']['checkpoints'][0]['runs']]
    for source in sources.values():
        assert [r['objective'] for r in source['checkpoints'][0]['runs']]==baseline
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    runs={name:[] for name in models}
    for action_seed in [72017,72018,72019,72020]:
        for name,model in models.items():
            result,p,f=rollout(s,w,model,action_seed);runs[name].append(result)
            print(action_seed,name,result['objective'],flush=True)
            np.savez_compressed(root/f'closed_loop_writer_validation_{name}_{action_seed}.npz',
                                prediction=p,failed=f,truth=w['truth'],video=w['video'],window_start=w['start'])
    comparison={}
    for name in sources:
        differences=np.array([a['objective']-b['objective'] for a,b in zip(runs[name],runs['zero'])])
        comparison[name]=dict(differences=differences.tolist(),mean=float(differences.mean()),
                              conditional_rng_se=float(differences.std(ddof=1)/2),
                              better_seeds=int((differences<0).sum()),selected_epoch=sources[name]['selected_epoch'])
        assert hashlib.sha256((root/f'closed_loop_writer_seed{name}.npz').read_bytes()).hexdigest()==hashes[name]
    result=dict(runs=runs,comparison=comparison,model_sha256=hashes,models_unchanged=True,
                baseline_checkpoint_exact=True,
                note='All3optimization seeds retained,no best-seed selection.4new actionseeds,same3reused TRAINhold videos,not independent-video confirmation. Fixed selectedmodels,no tuning/DEV/TEST/checker/default changes. SE conditional on these windows/model only.')
    (root/'closed_loop_writer_validation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
