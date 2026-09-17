"""Fresh TRAIN-holdout action RNG check of the already selected writer."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS
from continuous_residual_pilot import rollout


def main():
    root=Path('adaptive_search_results');source=json.loads((root/'continuous_residual_pilot.json').read_text())
    winner=source['selected_by_free_holdout']
    if winner=='zero':raise ValueError('No nonzero winner to validate')
    with np.load(root/source['model_files'][winner],allow_pickle=False) as a:model={k:a[k].copy() for k in a.files}
    model['kind']=str(model['kind'].item())
    seeds=list(range(22017,22025));assert not set(seeds)&{r['seed'] for r in source['holdout'][winner]}
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs=[]
    for seed in seeds:
        before,_,_=rollout(s,w,None,seed);after,_,_=rollout(s,w,model,seed)
        row=dict(seed=seed,baseline=before,candidate=after,objective_difference=after['objective']-before['objective'])
        runs.append(row);print(seed,row['objective_difference'],flush=True)
    differences=np.array([r['objective_difference'] for r in runs])
    report=dict(candidate=winner,runs=runs,mean_paired_difference=float(differences.mean()),
        conditional_rng_standard_error=float(differences.std(ddof=1)/np.sqrt(len(differences))),
        worse_seeds=int((differences>0).sum()),better_seeds=int((differences<0).sum()),
        note='Fixed already-selected model and same three TRAIN-holdout videos;8 fresh action RNG seeds, not new data or independent-video validation. Difference positive=worse. Standard error describes conditional RNG variation only. No parameter/threshold refit or new DEV/TEST use.')
    (root/'continuous_residual_gate_validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
