"""Fresh-seed comparison of frozen full/truncated pilot selections."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_writer import constant_model
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def load_models(root):
    models={'zero':None};selections={}
    for label,stem in [('full','hybrid_writer_pilot'),('truncated','hybrid_writer_truncated50_pilot')]:
        training=json.loads((root/f'{stem}.json').read_text())
        assert [r['seed'] for r in training['runs']]==[1901,2718,3141]
        for r in training['runs']:
            name=f'{label}_{r["seed"]}';models[name]=constant_model(np.array(r['selected_theta']),np.array(training['bound']))
            selections[name]=dict(epoch=r['selected_epoch'],theta=r['selected_theta'])
    return models,selections


def main():
    root=Path('adaptive_search_results');models,selections=load_models(root)
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);runs={name:[] for name in models}
    for seed in range(331017,331021):
        cache={}
        for name,model in models.items():
            key=(0.,0.) if model is None else tuple(model['value'])
            if key not in cache:cache[key]=rollout(s,w,model,seed)[0]
            runs[name].append(cache[key]);print(seed,name,cache[key]['objective'],flush=True)
    baseline=np.array([r['objective'] for r in runs['zero']]);summary={}
    for name,rows in runs.items():
        values=np.array([r['objective'] for r in rows]);delta=values-baseline
        summary[name]=dict(objective=float(values.mean()),delta=float(delta.mean()),conditional_seed_se=float(delta.std(ddof=1)/2),paired_delta=delta.tolist())
    report=dict(selections=selections,runs=runs,summary=summary,note='Frozen full and truncated selections,all optimization seeds retained;4new action seeds331017–20,same TRAIN-hold videos,not new independent videos. No reselection/tuning/DEV/TEST.')
    (root/'truncated_writer_validation.json').write_text(json.dumps(report,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
