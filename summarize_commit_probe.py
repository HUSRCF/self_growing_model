"""Matched strict-endpoint comparison of sparse, viability-only and full search."""
import json
from pathlib import Path
import numpy as np
from summarize_closed_loop import aggregate


def main():
    root=Path('adaptive_search_results');out=dict(modes={},alignment_verified=True,
       note='Same32DEV windows,3 rollout seeds; all3 TRAIN-selected optimizer checkpoints. No best-seed selection. All valid returned points include terminal-right validation. Probe is feasibility filtering, not future-score reranking.')
    modes=['sparse','probe','full']
    cache_paths=[root/f'adapted_probe_cache_model{m}.json' for m in [0,1901,2718,3141]]
    if all(p.exists() and len(json.loads(p.read_text())['runs'])==3 for p in cache_paths):modes.append('probe_cache')
    fast_paths=[root/f'adapted_probe_fast_model{m}.json' for m in [0,1901,2718,3141]]
    if all(p.exists() and len(json.loads(p.read_text())['runs'])==3 for p in fast_paths):modes.append('probe_fast')
    read_paths=[root/f'adapted_probe_read_model{m}.json' for m in [0,1901,2718,3141]]
    if all(p.exists() and len(json.loads(p.read_text())['runs'])==3 for p in read_paths):modes.append('probe_read')
    for mode in modes:
        docs=[json.loads((root/f'adapted_{mode}_model{m}.json').read_text()) for m in [0,1901,2718,3141]]
        dest=dict(per_model={});out['modes'][mode]=dest
        for d in docs:
            m=d['config']['model_seed'];runs=d['runs'];end_failures=0
            for r in runs:
                assert r['verification']['violations']==0
                seed=r['seed'];a=np.load(root/f'adapted_{mode}_model{m}_roll{seed}.npz')
                ref=np.load(root/f'adapted_sparse_model0_roll{seed}.npz')
                for k in ['truth','video','window_start']:np.testing.assert_array_equal(a[k],ref[k])
                if mode in ('probe_cache','probe_fast','probe_read'):
                    reference=np.load(root/f'adapted_probe_model{m}_roll{seed}.npz')
                    for k in ['prediction','failed','states','boundary']:np.testing.assert_array_equal(a[k],reference[k])
                    out['cache_bitwise_equal']=True
                    if mode=='probe_fast':
                        old=json.loads((root/f'adapted_probe_cache_model{m}.json').read_text())
                        old_run=next(x for x in old['runs'] if x['seed']==seed)
                        assert r['verification']==old_run['verification']
                        for k,v in r['stats'].items():
                            if k!='cpu_seconds':assert v==old_run['stats'][k],(m,seed,k)
                        out['fast_probe_bitwise_equal']=True
                        out['fast_probe_counters_equal']=True
                    if mode=='probe_read':
                        old=json.loads((root/f'adapted_probe_fast_model{m}.json').read_text())
                        old_run=next(x for x in old['runs'] if x['seed']==seed)
                        assert r['verification']==old_run['verification']
                        for k,v in r['stats'].items():
                            if k not in ('cpu_seconds','read_cache_hits','read_cache_misses'):
                                assert v==old_run['stats'][k],(m,seed,k)
                        assert r['stats']['read_cache_hits']+r['stats']['read_cache_misses']==r['stats']['expansions']
                        out['read_cache_bitwise_equal']=True
                        out['read_cache_search_counters_equal']=True
                end_failures+=int(a['failed'][:,:,-1].sum())
            dest['per_model'][str(m)]=dict(score=aggregate(runs),failed_endpoints=end_failures,total_particles=768,
                cpu_seconds=float(np.mean([r['stats']['cpu_seconds'] for r in runs])),
                expansions=float(np.mean([r['stats']['expansions'] for r in runs])),
                rollbacks=float(np.mean([r['stats']['rollbacks'] for r in runs])),
                probes=float(np.mean([r['stats'].get('viability_probes',0) for r in runs])),
                rejected_roots=float(np.mean([r['stats'].get('viability_rejected_roots',0) for r in runs])),
                full_probes=float(np.mean([r['stats'].get('viability_full_probes',0) for r in runs])),
                cache_hits=float(np.mean([r['stats'].get('rule_cache_hits',0) for r in runs])),
                cache_misses=float(np.mean([r['stats'].get('rule_cache_misses',0) for r in runs])),
                read_hits=float(np.mean([r['stats']['read_cache_hits'] for r in runs])) if all('read_cache_hits' in r['stats'] for r in runs) else None,
                read_misses=float(np.mean([r['stats']['read_cache_misses'] for r in runs])) if all('read_cache_misses' in r['stats'] for r in runs) else None)
        runs=[r for d in docs[1:] for r in d['runs']]
        dest['adapted_family']=dict(score=aggregate(runs),
            cpu_seconds=float(np.mean([r['stats']['cpu_seconds'] for r in runs])),
            failed_endpoints=sum(dest['per_model'][str(m)]['failed_endpoints'] for m in [1901,2718,3141]),
            total_particles=2304,
            per_video={v:{str(t):float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in runs]))
                          for t in [50,100,300]} for v in ['3','9','6','8']})
        print(mode,'base',dest['per_model']['0'],'family',dest['adapted_family'])
    (root/'commit_probe_summary.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
