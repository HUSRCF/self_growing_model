"""Fixed same-capacity constant and velocity writer family validation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');models={};sources={};paths={};hashes={}
    for family in ['constant','velocity']:
        for seed in [1901,2718,3141]:
            stem=f'closed_loop_writer_seed{seed}' if family=='constant' else f'closed_loop_writer_velocity_p4_i12_seed{seed}'
            name=f'{family}_{seed}';path=root/f'{stem}.npz';paths[name]=path
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:models[name]={k:z[k].copy() for k in z.files}
            sources[name]=json.loads((root/f'{stem}.json').read_text())
    for seed in [1901,2718,3141]:
        a=sources[f'constant_{seed}'];b=sources[f'velocity_{seed}']
        assert a['bound']==b['bound'] and a['training_particle_transitions']==b['training_particle_transitions']
        assert a['checkpoints'][0]['runs']==b['checkpoints'][0]['runs']
        for x,y in zip(a['trace'],b['trace']):
            for key in ['video','start','window_seed','action_seed']:assert x[key]==y[key]
        assert b['velocity_scale']==sources['velocity_1901']['velocity_scale']
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    runs={name:[] for name in ['zero',*models]};unique_evaluations=0
    for seed in [111017,111018,111019,111020]:
        baseline,_,_=rollout(s,w,None,seed);runs['zero'].append(baseline);cache={'zero':baseline};unique_evaluations+=1
        for name,model in models.items():
            key='zero' if np.all(model['value']==0) else (str(model['kind']),tuple(model['value']),tuple(model.get('velocity_scale',[])))
            if key not in cache:cache[key]=rollout(s,w,model,seed)[0];unique_evaluations+=1
            runs[name].append(cache[key])
        print(seed,{name:r[-1]['objective'] for name,r in runs.items()},flush=True)
    summary={}
    for family in ['constant','velocity']:
        names=[f'{family}_{seed}' for seed in [1901,2718,3141]]
        costs=np.array([[r['objective'] for r in runs[name]] for name in names])
        baseline=np.array([r['objective'] for r in runs['zero']]);delta=costs.mean(0)-baseline
        summary[family]=dict(objective=float(costs.mean()),mean_delta=float(delta.mean()),
            conditional_rng_se=float(delta.std(ddof=1)/2),per_action_seed_delta=delta.tolist(),
            selected_epochs=[sources[name]['selected_epoch'] for name in names],
            per_training_seed_delta=(costs-baseline).mean(1).tolist(),
            rmse={str(t):float(np.mean([r['score'][str(t)]['embedding_rmse'] for name in names for r in runs[name]])) for t in [50,100,300]})
    for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    report=dict(runs=runs,summary=summary,model_sha256=hashes,models_unchanged=True,
                matching_windows_budgets_bounds_verified=True,zero_checkpoint_exact=True,
                velocity_scale=sources['velocity_1901']['velocity_scale'],unique_evaluations=unique_evaluations,
                note='Same2parameters/maxbound/P4x12updates; effective RMS differs byvelocityfeature. All3optseeds retained,4newactionseeds/same3TRAINhold videos. Fixedmodels,no tuning/DEV/TEST/check enforcement. Family means not ensembleprediction; SE conditional on fixedvideos/models. Zero/identical parameters reuse same rollouts,not independent replicates.')
    (root/'velocity_writer_validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
