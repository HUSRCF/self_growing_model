"""Compare gradient accumulation at equal training samples and equal updates."""
import json
from pathlib import Path
import numpy as np
from summarize_closed_loop import aggregate


def main():
    root=Path('adaptive_search_results');seeds=[1901,2718,3141]
    groups={
      'single30':['closed_loop_policy','closed_loop_policy_train2718','closed_loop_policy_train3141'],
      'accum3_updates10':[f'accum3_updates10_seed{s}' for s in seeds],
      'accum3_updates30':[f'accum3_updates30_seed{s}' for s in seeds]}
    reference=json.loads((root/'closed_loop_policy.json').read_text())['arms']['frozen']
    result=dict(baseline=aggregate(reference),groups={},window_alignment=True,baseline_bitwise_equal=True,
      note='Same32DEV windows;3 optimizer seeds ×3 trajectory seeds per trained condition. No best-seed selection or significance claim.',
      budget_note='Equal budget refers to total training rollout batches, not updates in the selected checkpoint. All conditions have7 validation checkpoints but different sample positions.')
    for group,stems in groups.items():
        reports=[json.loads((root/(s+'.json')).read_text()) for s in stems]
        config=reports[0]['config'];updates=config['epochs'];accum=config.get('accumulate',1)
        result['groups'][group]=dict(updates=updates,batches_per_update=accum,
            total_training_batches=updates*accum,training_transitions_per_model=updates*accum*40*4*100,
            validation_checkpoints=7,arms={},per_optimizer={})
        dest=result['groups'][group]
        for seed,stem,report in zip(seeds,stems,reports):
            dest['per_optimizer'][str(seed)]={}
            for name in ['closed_loop','closed_loop_feedback']:
                t=report['training'][name]
                assert len(t['trace'])==7
                dest['per_optimizer'][str(seed)][name]=dict(selected_epoch=t['selected_epoch'],
                    selected_training_batches=t['selected_epoch']*accum,holdout_objective=t['holdout_objective'],
                    score=aggregate(report['arms'][name]))
            for r in report['arms']['frozen']:
                rs=r['seed'];ref=np.load(root/f'closed_loop_policy_frozen_{rs}.npz')
                raw=np.load(root/f'{stem}_frozen_{rs}.npz')
                for k in ['prediction','failed','truth','video','window_start']:
                    np.testing.assert_array_equal(ref[k],raw[k])
                for name in ['closed_loop','closed_loop_feedback']:
                    trial=np.load(root/f'{stem}_{name}_{rs}.npz')
                    for k in ['truth','video','window_start']:np.testing.assert_array_equal(ref[k],trial[k])
        for name in ['closed_loop','closed_loop_feedback']:
            runs=[r for report in reports for r in report['arms'][name]]
            dest['arms'][name]=dict(score=aggregate(runs),per_video={})
            for v in ['3','9','6','8']:
                dest['arms'][name]['per_video'][v]={str(t):float(np.mean(
                    [r['per_video'][v][str(t)]['embedding_rmse'] for r in runs])) for t in [50,100,300]}
    (root/'accumulation_summary.json').write_text(json.dumps(result,indent=2))
    for group,g in result['groups'].items():
        for name,a in g['arms'].items():
            print(group,name,[round(a['score'][str(t)]['embedding_rmse'],6) for t in [50,100,300]],
                  'energy3',round(a['score']['300']['energy_score'],6))


if __name__=='__main__':main()
