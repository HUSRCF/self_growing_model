"""All-seed, paired fresh-RNG validation of writer particle budgets."""
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
    for family in ['p4_i12','p12_i4','p12_i12']:
        for seed in [1901,2718,3141]:
            stem=f'closed_loop_writer_seed{seed}' if family=='p4_i12' else f'closed_loop_writer_{family}_seed{seed}'
            name=f'{family}_seed{seed}';path=root/f'{stem}.npz';paths[name]=path
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:models[name]={k:z[k].copy() for k in z.files}
            assert str(models[name]['kind'])=='constant'
            sources[name]=json.loads((root/f'{stem}.json').read_text())
    for seed in [1901,2718,3141]:
        assert sources[f'p12_i4_seed{seed}']['trace']==sources[f'p12_i12_seed{seed}']['trace'][:4]
    reference=sources['p4_i12_seed1901']['checkpoints'][0]['runs']
    for source in sources.values():assert source['checkpoints'][0]['runs']==reference
    replay=json.loads((root/'closed_loop_writer_seed1901_replay.json').read_text())
    assert replay['previous_exact_replay']
    s=AdaptiveBeam();w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    runs={name:[] for name in ['zero',*models]};unique_evaluations=0
    for action_seed in [101017,101018,101019,101020]:
        baseline,_,_=rollout(s,w,None,action_seed);runs['zero'].append(baseline)
        cache={(0.,0.):baseline};unique_evaluations+=1
        for name,model in models.items():
            key=tuple(model['value'].tolist())
            if key not in cache:
                cache[key]=rollout(s,w,model,action_seed)[0];unique_evaluations+=1
            runs[name].append(cache[key])
        print(action_seed,{name:r[-1]['objective'] for name,r in runs.items()},flush=True)
    summaries={}
    for family in ['p4_i12','p12_i4','p12_i12']:
        summaries[family]={}
        for seed in [1901,2718,3141]:
            name=f'{family}_seed{seed}';rows=runs[name]
            delta=np.array([r['objective']-b['objective'] for r,b in zip(rows,runs['zero'])])
            summaries[family][str(seed)]=dict(selected_epoch=sources[name]['selected_epoch'],
                selected_theta=sources[name]['selected_theta'],objective=float(np.mean([r['objective'] for r in rows])),
                paired_delta=delta.tolist(),mean_delta=float(delta.mean()),conditional_rng_se=float(delta.std(ddof=1)/2),
                better_action_seeds=int((delta<0).sum()),training_particle_transitions=sources[name]['training_particle_transitions'],
                distinct_training_windows=sources[name]['sampler']['distinct_sampled_windows'])
    for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    result=dict(runs=runs,summary=summaries,model_sha256=hashes,models_unchanged=True,
                first4_p12_updates_exact=True,default1901_exact_replay=True,common_zero_checkpoint_exact=True,
                unique_rollout_evaluations=unique_evaluations,
                note='All3optimization seeds per family, no bestseed picking. P4x12 andP12x4 equal simulatedtrainingtransitions;P12x12 triples them. Allsame4holdchecks but different timing/coverage/updatecount. Fixed selected models on4newactionseeds/same3reusedTRAINholdvideos. Identical constant vectors reused exactly within RNG, no new independent observations fromduplicates. No DEV/TEST/default promotion.')
    (root/'writer_budget_validation.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
