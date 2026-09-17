"""Frozen source-q versus shared corrections on TRAIN-hold prefix and tail."""
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_writer import constant_model
from train_closed_loop_policy import prefix_windows
from audit_writer_temporal_shift import select_videos
from confirm_truncated_writer import window_costs,paired
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import tail_windows


def main():
    root=Path('adaptive_search_results');models={'zero':None};selections={}
    for label,stem in [('shared','hybrid_writer_truncated50_pilot'),('source_q','hybrid_writer_source_q_pilot')]:
        training=json.loads((root/f'{stem}.json').read_text())
        assert [r['seed'] for r in training['runs']]==[1901,2718,3141]
        for r in training['runs']:
            name=f'{label}_{r["seed"]}';m=constant_model(np.array(r['selected_theta']),np.array(training['bound']))
            if label=='source_q':m['kind']='source_q'
            models[name]=m;selections[name]=dict(epoch=r['selected_epoch'],theta=r['selected_theta'])
    s=AdaptiveBeam();regions={'prefix':prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8),
                            'tail':select_videos(tail_windows('train',300,8),SPLITS['train'][-3:])}
    allruns={};summary={}
    for region,w in regions.items():
        runs={name:[] for name in models}
        for seed in range(361017,361021):
            cache={}
            for name,model in models.items():
                values=np.zeros((8,2)) if model is None else np.broadcast_to(model['value'],(8,2))
                key=tuple(values.reshape(-1))
                if key not in cache:
                    row,p,f=rollout(s,w,model,seed);c=window_costs(p,w['truth'],f)
                    np.testing.assert_allclose(c.mean(),row['objective'],rtol=0,atol=1e-14)
                    row['per_video_u_cost']={str(v):float(c[w['video']==v].mean()) for v in np.unique(w['video'])}
                    cache[key]=row
                runs[name].append(cache[key]);print(region,seed,name,cache[key]['objective'],flush=True)
        base=[r['objective'] for r in runs['zero']];summary[region]={}
        for name,rows in runs.items():
            summary[region][name]=paired([r['objective'] for r in rows],base)
            summary[region][name]['per_video']={v:paired([r['per_video_u_cost'][v] for r in rows],[r['per_video_u_cost'][v] for r in runs['zero']]) for v in rows[0]['per_video_u_cost']}
        allruns[region]=runs
    report=dict(selections=selections,runs=allruns,summary=summary,
                note='All source-q/shared selections frozen. Four new action seeds361017–20,TRAINhold3videos prefix and tail each24windows/P8/300. No retraining/reselection/DEV/TEST. Same seeds across regions do not isolate causal temporal effects. Conditional RNG SE only.')
    (root/'source_q_writer_validation.json').write_text(json.dumps(report,indent=2))
    print({region:{name:(v['objective'],v['delta'],v['better']) for name,v in values.items()} for region,values in summary.items()},flush=True)


if __name__=='__main__':main()
