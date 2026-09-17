"""Fixed checkpoint selection audit; cross-evaluate independent RNG halves."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from train_closed_loop_writer import constant_model
from v20_rnn_mixture.engine.common import SPLITS


def summarize_checkpoints(costs,selected):
    c=np.asarray(costs);delta=c-c[:,:1]
    choices=[int(np.argmin(c[i:i+4].mean(0))) for i in [0,4]]
    cross=[float(delta[4:,choices[0]].mean()),float(delta[:4,choices[1]].mean())]
    return dict(mean_costs=c.mean(0).tolist(),mean_deltas=delta.mean(0).tolist(),
                conditional_rng_se=(delta.std(0,ddof=1)/np.sqrt(len(c))).tolist(),
                original_selected_index=int(selected),original_selected_delta=float(delta[:,selected].mean()),
                fresh_mean_best_index=int(np.argmin(c.mean(0))),half_selected_indices=choices,
                half_cross_deltas=cross,mean_cross_delta=float(np.mean(cross)),
                original_selected_better_seeds=int((delta[:,selected]<0).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['constant','velocity'],required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    root=Path('adaptive_search_results')
    stem=f'closed_loop_writer_seed{args.seed}' if args.family=='constant' else f'closed_loop_writer_velocity_p4_i12_seed{args.seed}'
    path=root/f'{stem}.json';modelpath=root/f'{stem}.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest();modeldigest=hashlib.sha256(modelpath.read_bytes()).hexdigest()
    source=json.loads(path.read_text());models=[]
    for cp in source['checkpoints']:
        m=constant_model(np.array(cp['theta']),np.array(source['bound']))
        if args.family=='velocity':m.update(kind='velocity',velocity_scale=np.array(source['velocity_scale']))
        models.append(m)
    epochs=[c['epoch'] for c in source['checkpoints']];assert epochs==[0,4,8,12]
    chosen=epochs.index(source['selected_epoch'])
    with np.load(modelpath) as stored:
        for key,value in models[chosen].items():np.testing.assert_array_equal(value,stored[key])
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8);costs=[];counts=0
    for action_seed in [62017,*range(121017,121025)]:
        values=[];cache={}
        for index,m in enumerate(models):
            key=tuple(m['value'].tolist())
            if key not in cache:cache[key]=rollout(s,w,m,action_seed)[0];counts+=1
            r=cache[key];values.append(r['objective'])
            if action_seed==62017:assert r==source['checkpoints'][index]['runs'][0]
        if action_seed!=62017:costs.append(values)
        print(args.family,args.seed,action_seed,values,flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==modeldigest
    report=dict(family=args.family,optimization_seed=args.seed,epochs=epochs,costs=costs,
                action_seeds=list(range(121017,121025)),summary=summarize_checkpoints(costs,chosen),
                source_sha256=digest,model_sha256=modeldigest,models_unchanged=True,
                original62017_exact_replay=True,selected_parameters_exact=True,unique_evaluations=counts,
                note='Frozen4checkpoints,8freshRNG on same3TRAINholdvideos. Four-RNG half chooses,otherhalf evaluates; reverse too. All8best is optimistic diagnostic only, not deployment selection. No refit/DEV/TEST. Identical checkpoints reuse evaluation; not independent observations.')
    (root/f'writer_checkpoint_audit_{args.family}_{args.seed}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
