"""Matched history/selection and simulated-step horizon controls."""
import json
from pathlib import Path
import numpy as np
from summarize_objective_training import aggregate
from summarize_particle_frontier import energy_u


def main():
    root=Path('adaptive_search_results');out=dict(groups={});baseline=None
    for kind in ['short','long','short_budget']:
        docs=[];per_model={}
        for m in [1901,2718,3141]:
            stem=f'horizon_{kind}_seed{m}';d=json.loads((root/f'{stem}.json').read_text());docs.append(d)
            short=json.loads((root/f'horizon_short_seed{m}.json').read_text())
            for arm,tr in d['training'].items():
                assert len(tr['train_trace'])==30 and len(tr['trace'])==7
                assert tr['simulated_training_steps']==(480000 if kind=='short' else 1440000)
                assert tr['sampler_audit']['per_video']==short['training'][arm]['sampler_audit']['per_video']
                if kind!='short_budget':assert tr['sampler_audit']==short['training'][arm]['sampler_audit']
                assert tr['trace'][0]['holdout_objective']==short['training'][arm]['trace'][0]['holdout_objective']
                assert tr['holdout_objective']<=tr['trace'][0]['holdout_objective']
            for arm,runs in d['arms'].items():
                assert len(runs)==3
                for r in runs:
                    seed=r['seed']
                    with np.load(root/f'{stem}_{arm}_{seed}.npz') as a,np.load(root/f'feedback_distribution_pilot_frozen_point_{seed}.npz') as b:
                        for k in ['truth','video','window_start']:np.testing.assert_array_equal(a[k],b[k])
                        if arm=='frozen' or d['training'][arm]['selected_epoch']==0:
                            for k in ['prediction','failed']:np.testing.assert_array_equal(a[k],b[k])
                        pred=a['prediction'];truth=a['truth'];p=pred.shape[1]
                        for t in [50,100,300]:
                            x=pred[:,:,t-1];y=truth[:,t-1]
                            emb=np.concatenate([np.sin(x),np.cos(x)],-1);target=np.concatenate([np.sin(y),np.cos(y)],-1)
                            distance=np.linalg.norm(emb-target[:,None],axis=-1).mean(1)
                            np.testing.assert_allclose(np.sqrt(((emb.mean(1)-target)**2).mean()),r['score'][str(t)]['embedding_rmse'],atol=1e-14,rtol=1e-14)
                            r['score'][str(t)]['energy_score_u']=energy_u(r['score'][str(t)]['energy_score'],distance.mean(),p)
                            for v in np.unique(a['video']):
                                s=r['per_video'][str(v)][str(t)]
                                s['energy_score_u']=energy_u(s['energy_score'],distance[a['video']==v].mean(),p)
            if baseline is None:baseline=d['arms']['frozen']
            per_model[str(m)]={arm:dict(score=aggregate(d['arms'][arm]),epoch=tr['selected_epoch'],holdout=tr['holdout_objective'],
                initial_holdout=tr['trace'][0]['holdout_objective'],simulated_steps=tr['simulated_training_steps'],
                gradient_norm_max=max(t['gradient_norm'] for t in tr['train_trace']),
                clipped_updates=sum(t['gradient_norm']>1 for t in tr['train_trace'])) for arm,tr in d['training'].items()}
        out['groups'][kind]=dict(per_model=per_model,arms={})
        for arm in ['closed_loop','closed_loop_feedback']:
            runs=[r for d in docs for r in d['arms'][arm]]
            result=dict(score=aggregate(runs),per_video={v:aggregate([dict(score=r['per_video'][v]) for r in runs]) for v in ['3','9','6','8']})
            out['groups'][kind]['arms'][arm]=result
            print(kind,arm,[round(result['score'][str(t)]['embedding_rmse'],6) for t in [50,100,300]],
                  'energyU',[round(result['score'][str(t)]['energy_score_u'],6) for t in [50,100,300]],flush=True)
    replay=json.loads((root/'horizon_default_parity_seed1901.json').read_text())
    previous=json.loads((root/'objective_mse_seed1901.json').read_text())
    for arm,tr in replay['training'].items():
        assert tr['train_trace']==previous['training'][arm]['train_trace']
        assert tr['selected_epoch']==previous['training'][arm]['selected_epoch']
        with np.load(root/f'horizon_default_parity_seed1901_{arm}.npz') as a,np.load(root/f'objective_mse_seed1901_{arm}.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    for arm,runs in replay['arms'].items():
        for r in runs:
            seed=r['seed']
            with np.load(root/f'horizon_default_parity_seed1901_{arm}_{seed}.npz') as a,np.load(root/f'objective_mse_seed1901_{arm}_{seed}.npz') as b:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    out.update(frozen=aggregate(baseline),default_100_step_training_and_predictions_exact=True,
        common_300_eligible_pool_verified=True,common_selection_baseline_verified=True,
        note='TRAIN-prefix only, same300-eligible histories; same300-step holdout selection (mean energy at50/100/300). Short vs long matches30updates/batches; short_budget vs long matches1.44M simulated training transitions/model but not histories per update. Loss divided by rollout length, so global policy-gradient scale differs across100/300. Fixed KL and Adam. No checker/search; numeric completion is not constraint validity. All3 optimizer seeds,3 rollout seeds, same32DEV windows, no TEST or significance claim.')
    (root/'training_horizon_summary.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
