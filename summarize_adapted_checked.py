"""Compare adapter families only against their matching runtime baseline."""
import json
from pathlib import Path
import numpy as np
from summarize_closed_loop import aggregate


def main():
    root=Path('adaptive_search_results');out=dict(runtimes={},window_alignment=True,
        note='DEV32,3 optimizer ×3 trajectory seeds for adapter. No best-seed selection. Checked and sparse differ in budgets/sampling; matched baseline within each. All valid outputs independently centrally checked, including endpoint.')
    for mode in ['checked','sparse']:
        docs=[json.loads((root/f'adapted_{mode}_model{m}.json').read_text()) for m in [0,1901,2718,3141]]
        baseline=docs[0]['runs'];adapted=[r for d in docs[1:] for r in d['runs']]
        dest=dict(baseline=aggregate(baseline),adapted=aggregate(adapted),per_optimizer={},per_video={},
                  baseline_cpu=float(np.mean([r['stats']['cpu_seconds'] for r in baseline])),
                  adapted_cpu=float(np.mean([r['stats']['cpu_seconds'] for r in adapted])),
                  baseline_rollbacks=float(np.mean([r['stats']['rollbacks'] for r in baseline])),
                  adapted_rollbacks=float(np.mean([r['stats']['rollbacks'] for r in adapted])))
        out['runtimes'][mode]=dest
        for d in docs:
            m=d['config']['model_seed']
            if m:dest['per_optimizer'][str(m)]=aggregate(d['runs'])
            for r in d['runs']:
                assert r['verification']['violations']==0
                seed=r['seed'];a=np.load(root/f'adapted_{mode}_model{m}_roll{seed}.npz')
                ref=np.load(root/f'adapted_{mode}_model0_roll{seed}.npz')
                for key in ['truth','video','window_start']:np.testing.assert_array_equal(a[key],ref[key])
        for v in ['3','9','6','8']:
            dest['per_video'][v]={str(t):dict(baseline=float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in baseline])),
                adapted=float(np.mean([r['per_video'][v][str(t)]['embedding_rmse'] for r in adapted]))) for t in [50,100,300]}
        print(mode,'base',dest['baseline'],'adapted',dest['adapted'],'cpu',dest['baseline_cpu'],dest['adapted_cpu'])
    (root/'adapted_checked_summary.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
