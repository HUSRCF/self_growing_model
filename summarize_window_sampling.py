"""Matched-budget initial-state coverage and prediction audit."""
import json
from pathlib import Path
import numpy as np
from summarize_closed_loop import aggregate


def main():
    root=Path('adaptive_search_results');seeds=[1901,2718,3141]
    stems={'fixed':['closed_loop_policy','closed_loop_policy_train2718','closed_loop_policy_train3141'],
           **{mode:[f'windows_{mode}_seed{s}' for s in seeds] for mode in ['uniform','stratified']}}
    baseline=json.loads((root/'closed_loop_policy.json').read_text())['arms']['frozen']
    result=dict(baseline=aggregate(baseline),conditions={},baseline_bitwise_equal=True,window_alignment=True,
      note='Same32DEV windows,3 optimizer ×3 trajectory seeds. No independent-video replication or significance claim. All30 training batches/updates,7 selection checkpoints.')
    for mode,paths in stems.items():
        docs=[json.loads((root/(p+'.json')).read_text()) for p in paths]
        dest=dict(arms={},per_optimizer={});result['conditions'][mode]=dest
        for seed,path,d in zip(seeds,paths,docs):
            dest['per_optimizer'][str(seed)]={}
            for arm in ['closed_loop','closed_loop_feedback']:
                tr=d['training'][arm]
                dest['per_optimizer'][str(seed)][arm]=dict(selected_epoch=tr['selected_epoch'],
                     score=aggregate(d['arms'][arm]),sampler_audit=tr.get('sampler_audit'))
            if mode!='fixed':
                assert d['training']['closed_loop']['sampler_audit']==d['training']['closed_loop_feedback']['sampler_audit']
            for r in d['arms']['frozen']:
                seed_r=r['seed'];ref=np.load(root/f'closed_loop_policy_frozen_{seed_r}.npz')
                trial=np.load(root/f'{path}_frozen_{seed_r}.npz')
                for k in ['prediction','failed','truth','video','window_start']:np.testing.assert_array_equal(ref[k],trial[k])
                for arm in ['closed_loop','closed_loop_feedback']:
                    trial=np.load(root/f'{path}_{arm}_{seed_r}.npz')
                    for k in ['truth','video','window_start']:np.testing.assert_array_equal(ref[k],trial[k])
                    if d['training'][arm]['selected_epoch']==0:np.testing.assert_array_equal(ref['prediction'],trial['prediction'])
        for arm in ['closed_loop','closed_loop_feedback']:
            runs=[r for d in docs for r in d['arms'][arm]]
            dest['arms'][arm]=dict(score=aggregate(runs),per_video={})
            for v in ['3','9','6','8']:
                dest['arms'][arm]['per_video'][v]={str(t):float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in runs]))
                                                  for t in [50,100,300]}
            print(mode,arm,[round(dest['arms'][arm]['score'][str(t)]['embedding_rmse'],6) for t in [50,100,300]],
                  'energy3',round(dest['arms'][arm]['score']['300']['energy_score'],6))
    expanded_paths=[root/f'refreshed128_model{s}.json' for s in [0,1901,2718,3141]]
    if all(p.exists() for p in expanded_paths):
        expanded=[json.loads(p.read_text()) for p in expanded_paths]
        all_runs=[r for d in expanded[1:] for r in d['runs']]
        result['denser_dev']=dict(baseline=aggregate(expanded[0]['runs']),
            uniform=aggregate(all_runs),per_optimizer={str(d['model_seed']):aggregate(d['runs']) for d in expanded[1:]},
            per_video={},note=expanded[0]['note'])
        for v in ['3','9','6','8']:
            result['denser_dev']['per_video'][v]={str(t):dict(
                baseline=float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in expanded[0]['runs']])),
                uniform=float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in all_runs]))) for t in [50,100,300]}
        for d in expanded[1:]:
            for r in d['runs']:
                rs=r['seed'];model=d['model_seed']
                ref=np.load(root/f'refreshed128_model0_roll{rs}.npz')
                trial=np.load(root/f'refreshed128_model{model}_roll{rs}.npz')
                for k in ['truth','video','window_start']:np.testing.assert_array_equal(ref[k],trial[k])
        print('denser128',result['denser_dev']['baseline'],result['denser_dev']['uniform'])
    (root/'window_sampling_summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
