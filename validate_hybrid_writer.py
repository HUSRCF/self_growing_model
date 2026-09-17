"""Freeze all three TRAIN-selected corrections and evaluate fresh action seeds."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_writer import constant_model
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');training=json.loads((root/'hybrid_writer_pilot.json').read_text())
    assert [r['seed'] for r in training['runs']]==[1901,2718,3141]
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    models={'zero':None}
    for r in training['runs']:models[str(r['seed'])]=constant_model(np.array(r['selected_theta']),np.array(training['bound']))
    runs={name:[] for name in models}
    for seed in [311017,311018,311019,311020]:
        cache={}
        for name,model in models.items():
            key=(0.,0.) if model is None else tuple(model['value'])
            if key not in cache:cache[key]=rollout(s,w,model,seed)[0]
            runs[name].append(cache[key]);print(seed,name,cache[key]['objective'],flush=True)
    baseline=np.array([r['objective'] for r in runs['zero']]);summary={}
    for name,rows in runs.items():
        values=np.array([r['objective'] for r in rows]);delta=values-baseline
        summary[name]=dict(objective=float(values.mean()),delta=float(delta.mean()),conditional_seed_se=float(delta.std(ddof=1)/2),paired_delta=delta.tolist())
    report=dict(runs=runs,summary=summary,note='Frozen selections; four fresh action seeds on SAME TRAIN-hold videos, not independent video generalization. No refitting/reselection/DEV/TEST. All three optimization seeds retained.')
    (root/'hybrid_writer_validation.json').write_text(json.dumps(report,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
