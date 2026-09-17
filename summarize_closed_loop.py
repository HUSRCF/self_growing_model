"""Paired audit of three optimizer seeds, not nine independent datasets."""
import json
from pathlib import Path
import numpy as np


def aggregate(runs):
    return {str(t):{metric:float(np.mean([r['score'][str(t)][metric] for r in runs]))
                   for metric in ['embedding_rmse','energy_score','coverage90','failure']}
            for t in [50,100,300]}


def main():
    root=Path('adaptive_search_results')
    stems=['closed_loop_policy','closed_loop_policy_train2718','closed_loop_policy_train3141']
    reports=[json.loads((root/(stem+'.json')).read_text()) for stem in stems]
    reference=reports[0]['arms']['frozen']
    output=dict(optimizer_seeds=[1901,2718,3141],trajectory_seeds=[1729,2718,3141],
                baseline=aggregate(reference),arms={},per_optimizer={},
                note='Means over optimizer/trajectory seeds on the SAME32DEV windows. Not nine independent datasets; no significance claim.',
                baseline_bitwise_equal=True,window_alignment=True)
    for stem,report in zip(stems,reports):
        seed=report['config'].get('train_seed',1901)
        output['per_optimizer'][str(seed)]={}
        for name in ['closed_loop','closed_loop_feedback']:
            output['per_optimizer'][str(seed)][name]=dict(
                selected_epoch=report['training'][name]['selected_epoch'],
                holdout_objective=report['training'][name]['holdout_objective'],
                score=aggregate(report['arms'][name]))
        for r in report['arms']['frozen']:
            rollseed=r['seed']
            ref=np.load(root/f'feedback_distribution_pilot_frozen_point_{rollseed}.npz')
            baseline=np.load(root/f'{stem}_frozen_{rollseed}.npz')
            for key in ['prediction','failed','truth','video','window_start']:
                np.testing.assert_array_equal(ref[key],baseline[key])
            for name in ['closed_loop','closed_loop_feedback']:
                trial=np.load(root/f'{stem}_{name}_{rollseed}.npz')
                for key in ['truth','video','window_start']:
                    np.testing.assert_array_equal(ref[key],trial[key])
    for name in ['closed_loop','closed_loop_feedback']:
        runs=[r for report in reports for r in report['arms'][name]]
        output['arms'][name]=dict(score=aggregate(runs),per_video={})
        for video in ['3','9','6','8']:
            output['arms'][name]['per_video'][video]={str(t):dict(
                rmse=float(np.mean([r['per_video'][video][str(t)]['embedding_rmse'] for r in runs])),
                baseline_rmse=float(np.mean([r['per_video'][video][str(t)]['embedding_rmse'] for r in reference])))
                for t in [50,100,300]}
    (root/'closed_loop_summary.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
