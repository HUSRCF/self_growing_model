"""Aggregate fixed-checkpoint audit without selecting deployment models."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results');rows=[];families={};baseline_reference=None
    for family in ['constant','velocity']:
        original=[];cross=[];optimistic=[]
        for seed in [1901,2718,3141]:
            r=json.loads((root/f'writer_checkpoint_audit_{family}_{seed}.json').read_text())
            s=r['summary'];c=np.array(r['costs']);delta=c-c[:,:1]
            if baseline_reference is None:baseline_reference=c[:,0].copy()
            np.testing.assert_array_equal(c[:,0],baseline_reference)
            selected=s['original_selected_index']
            original.append(delta[:,selected]);cross.append(s['half_cross_deltas'])
            optimistic.append(float(delta.mean(0).min()))
            rows.append(dict(family=family,seed=seed,original_epoch=r['epochs'][selected],
                fresh_mean_best_epoch=r['epochs'][s['fresh_mean_best_index']],
                half_chosen_epochs=[r['epochs'][i] for i in s['half_selected_indices']],
                original_delta=s['original_selected_delta'],cross_delta=s['mean_cross_delta'],
                checkpoint_mean_deltas=s['mean_deltas'],
                nonzero_checkpoints_improving_both_halves=int(((delta[:4,1:].mean(0)<0)&(delta[4:,1:].mean(0)<0)).sum())))
        original=np.mean(original,axis=0)
        families[family]=dict(original_selected_mean_delta=float(original.mean()),
            original_selected_conditional_rng_se=float(original.std(ddof=1)/np.sqrt(8)),
            half_cross_selected_mean_delta=float(np.mean(cross)),
            optimistic_all8_selected_mean_delta=float(np.mean(optimistic)))
    out=dict(rows=rows,families=families,
        original_matches_all8_best=sum(r['original_epoch']==r['fresh_mean_best_epoch'] for r in rows),
        half_winner_agreement=sum(r['half_chosen_epochs'][0]==r['half_chosen_epochs'][1] for r in rows),
        note='Originalselection uses prior2RNG,halfselection uses4RNG thenother4forassessment. All8min is optimistic diagnostic, not demonstrated achievable performance.6runs share videos and RNG; familySE averagesmodels first,not independentvideo inference. No model changes.')
    (root/'writer_checkpoint_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
