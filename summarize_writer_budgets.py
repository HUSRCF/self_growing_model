"""Family-level paired summaries; training seeds are not independent videos."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results');report=json.loads((root/'writer_budget_validation.json').read_text())
    runs=report['runs'];out={}
    baseline=np.array([r['objective'] for r in runs['zero']])
    for family in ['p4_i12','p12_i4','p12_i12']:
        names=[f'{family}_seed{s}' for s in [1901,2718,3141]]
        values=np.array([[r['objective'] for r in runs[name]] for name in names])
        delta=values.mean(0)-baseline
        out[family]=dict(objective=float(values.mean()),paired_mean_delta=float(delta.mean()),
            per_action_seed_family_delta=delta.tolist(),conditional_rng_se=float(delta.std(ddof=1)/2),
            rmse={str(t):float(np.mean([r['score'][str(t)]['embedding_rmse'] for name in names for r in runs[name]])) for t in [50,100,300]},
            selected_epochs=[report['summary'][family][str(s)]['selected_epoch'] for s in [1901,2718,3141]],
            per_video_rmse3={str(v):float(np.mean([r['per_video'][str(v)]['300']['embedding_rmse'] for name in names for r in runs[name]])) for v in [18,16,13]},
            max_numeric_failure=max(r['score']['300']['failure'] for name in names for r in runs[name]))
    result=dict(families=out,baseline_objective=float(baseline.mean()),
                baseline_rmse={str(t):float(np.mean([r['score'][str(t)]['embedding_rmse'] for r in runs['zero']])) for t in [50,100,300]},
                note='Family means average performance of3fixedtrained models, not prediction ensemble. SE over4actionseeds after averaging models, conditional on same3holdvideos; no12-independent-sample claim. Identical selectedparameters may occur acrossfamilies.')
    (root/'writer_budget_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
