"""Common-window sparse-search ablation summaries (DEV only)."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results'); reference=np.load(root/'diversity32_t1_s1729.npz')
    report={}
    for name in ['shallow','period5','guard5','shallow_t2','period5_matched','full']:
        runs=[]
        for seed in [1729,2718,3141]:
            p=root/(f'diversity32_t1_s{seed}.json' if name=='full' else f'sparse32_{name}_s{seed}.json')
            runs.append(json.loads(p.read_text()))
            a=np.load(p.with_suffix('.npz'))
            for k in ['history','truth','video','window_start']:
                np.testing.assert_array_equal(reference[k],a[k])
        report[name]=dict(metrics={h:{k:float(np.mean([r['score'][h][k] for r in runs]))
                                      for k in ['embedding_rmse','energy_score','coverage90','failure']}
                                  for h in ['50','100','300']},
                          rollbacks=float(np.mean([r['rollbacks'] for r in runs])),
                          expansions=float(np.mean([r['audit']['expansions'] for r in runs])),
                          seconds=[r['seconds'] for r in runs],
                          mean_depth=float(np.mean([r['search']['mean_depth'] for r in runs])))
    out=dict(results=report,note='32 DEV windows,8 particles,3 seeds. Same windows verified. Unequal workers/concurrent load: compare numerical expansion counts, not wall time as an equal-budget benchmark. Depth-normalized scores confound sampling temperature; matched variants control root-prior coefficient only.')
    (root/'sparse32_summary.json').write_text(json.dumps(out,indent=2))
    for name,r in report.items():
        print(name,{h:round(r['metrics'][h]['embedding_rmse'],6) for h in ['50','100','300']},
              'expansions',r['expansions'],'rollbacks',r['rollbacks'])


if __name__=='__main__':main()
