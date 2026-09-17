"""Objective versus checkpoint-selection controls, all optimizer seeds."""
import json
from pathlib import Path
import numpy as np
from summarize_particle_frontier import energy_u


def aggregate(runs):
    return {str(t):{k:float(np.mean([r['score'][str(t)][k] for r in runs]))
                   for k in ['embedding_rmse','energy_score','energy_score_u','coverage90','failure']}
            for t in [50,100,300]}


def main():
    root=Path('adaptive_search_results');out=dict(groups={},checks={});reference_runs=None
    for kind in ['mse','energy_u','mse_select_energy']:
        docs=[];per_model={}
        for m in [1901,2718,3141]:
            stem=f'objective_{kind}_seed{m}';d=json.loads((root/f'{stem}.json').read_text());docs.append(d)
            for arm,train in d['training'].items():
                assert len(train['train_trace'])==30 and train['train_trace'][-1]['batches_seen']==30
                assert len(train['trace'])==7
                assert train['holdout_objective']<=train['trace'][0]['holdout_objective']
                old=json.loads((root/f'objective_mse_seed{m}.json').read_text())['training'][arm]
                assert train['sampler_audit']==old['sampler_audit']
                if kind=='mse_select_energy':assert train['train_trace']==old['train_trace']
                if kind=='mse':
                    previous=json.loads((root/f'windows_uniform_seed{m}.json').read_text())['training'][arm]
                    assert train['train_trace']==previous['train_trace']
                    assert train['selected_epoch']==previous['selected_epoch']
                    with np.load(root/f'{stem}_{arm}.npz') as a,np.load(root/f'windows_uniform_seed{m}_{arm}.npz') as b:
                        assert set(a.files)==set(b.files)
                        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
            for arm,runs in d['arms'].items():
                assert len(runs)==3
                for r in runs:
                    seed=r['seed']
                    with np.load(root/f'{stem}_{arm}_{seed}.npz') as a,np.load(root/f'feedback_distribution_pilot_frozen_point_{seed}.npz') as base:
                        for k in ['truth','video','window_start']:np.testing.assert_array_equal(a[k],base[k])
                        if arm=='frozen' or d['training'][arm]['selected_epoch']==0:
                            for k in ['prediction','failed']:np.testing.assert_array_equal(a[k],base[k])
                        if kind=='mse':
                            with np.load(root/f'windows_uniform_seed{m}_{arm}_{seed}.npz') as old:
                                for k in ['prediction','failed']:np.testing.assert_array_equal(a[k],old[k])
                        pred=a['prediction'];truth=a['truth'];p=pred.shape[1]
                        for t in [50,100,300]:
                            x=pred[:,:,t-1];y=truth[:,t-1]
                            emb=np.concatenate([np.sin(x),np.cos(x)],-1);target=np.concatenate([np.sin(y),np.cos(y)],-1)
                            distance=np.linalg.norm(emb-target[:,None],axis=-1).mean(1)
                            r['score'][str(t)]['energy_score_u']=energy_u(r['score'][str(t)]['energy_score'],float(distance.mean()),p)
                            for v in np.unique(a['video']):
                                s=r['per_video'][str(v)][str(t)]
                                s['energy_score_u']=energy_u(s['energy_score'],float(distance[a['video']==v].mean()),p)
            if reference_runs is None:reference_runs=d['arms']['frozen']
            per_model[str(m)]={arm:dict(score=aggregate(d['arms'][arm]),selected_epoch=train['selected_epoch'],
                selected_holdout=train['holdout_objective'],initial_holdout=train['trace'][0]['holdout_objective'],
                gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in train['train_trace']])),
                gradient_norm_max=max(r['gradient_norm'] for r in train['train_trace']),
                clipped_updates=sum(r['gradient_norm']>1 for r in train['train_trace'])) for arm,train in d['training'].items()}
        out['groups'][kind]=dict(per_model=per_model,arms={})
        for arm in ['closed_loop','closed_loop_feedback']:
            runs=[r for d in docs for r in d['arms'][arm]]
            out['groups'][kind]['arms'][arm]=dict(score=aggregate(runs),
                per_video={v:aggregate([dict(score=r['per_video'][v]) for r in runs]) for v in ['3','9','6','8']})
            print(kind,arm,[round(aggregate(runs)[str(t)]['embedding_rmse'],6) for t in [50,100,300]],
                  'energyU',[round(aggregate(runs)[str(t)]['energy_score_u'],6) for t in [50,100,300]],flush=True)
    out.update(frozen=aggregate(reference_runs),checks=dict(mse_training_and_predictions_exact_previous=True,
          selection_control_training_trace_exact=True,all_sampler_audits_equal=True,frozen_and_zero_epoch_predictions_exact=True,window_alignment=True),
          note='All3 optimizer seeds×3 rollout seeds on same32DEV windows. TRAIN-prefix fitting,7 TRAIN-video-holdout checkpoints including zero. First compare mse/mse vs energy/energy as whole pipelines; mse/energy control separates selection from training objective. Fixed Adam/gradient scale but different loss units; see gradient norms/clipping. No checker/search in any training/evaluation arm: numeric completion is not constraint validity. 3s extrapolates beyond100 training steps. Not blind or independent-video replication.')
    (root/'objective_training_summary.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
