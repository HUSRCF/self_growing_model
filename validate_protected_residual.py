"""Additional RNG validation; fixed protected writer, same TRAIN hold videos."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');path=root/'protected_residual_model.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    source=json.loads((root/'protected_residual_pilot.json').read_text())
    seeds=list(range(52017,52025))
    assert not set(seeds)&{r['seed'] for r in source['holdout']['protected']}
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs=[]
    for seed in seeds:
        before,_,_=rollout(s,w,None,seed);after,_,_=rollout(s,w,model,seed)
        runs.append(dict(seed=seed,baseline=before,candidate=after,
                         objective_difference=after['objective']-before['objective']))
        print(seed,runs[-1]['objective_difference'],flush=True)
    differences=np.array([r['objective_difference'] for r in runs])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    report=dict(alpha=source['alpha'],model_sha256=digest,model_unchanged=True,runs=runs,
                mean_paired_difference=float(differences.mean()),
                conditional_rng_standard_error=float(differences.std(ddof=1)/np.sqrt(len(differences))),
                better_seeds=int((differences<0).sum()),worse_seeds=int((differences>0).sum()),
                note='Eight additional action seeds, same three reused TRAIN hold videos. Fixed alpha/model, no tuning, DEV/TEST or deployment. Positive difference=worse. SE measures conditional RNG variation, not independent-video generalization.')
    (root/'protected_residual_validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
