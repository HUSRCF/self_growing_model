"""Cached historical holdout screen of frozen group-risk candidates, no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from confirm_temporal_kernel_holdout import hold_windows
from validate_soft_gates_new_windows import causal_features
from validate_group_risk_gate import predictions
from soft_output_mixture import value_gradient,interval_grid_difference


def main():
    root=Path('adaptive_search_results')
    paths=[root/'group_risk_gate_model.json',root/'temporal_kernel_holdout.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model=json.loads(paths[0].read_text());source=json.loads(paths[1].read_text())
    assert source['all_precision_gates_pass']
    for parent in [model,source]:
        for p,h in parent['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    engine=AdaptiveBeam();runs=[];summary={}
    for region in ['prefix','tail']:
        w=hold_windows(region);motion,_=causal_features(engine,w)
        alpha=predictions(model['models'],model['reference'],motion)
        selected=sorted([r for r in source['runs'] if r['region']==region and r['kernel']=='new'],key=lambda r:r['seed'])
        for row in selected:
            metadata=next(r for r in source['metadata'] if r['region']==region and r['seed']==row['seed'])
            np.testing.assert_array_equal(w['video'],metadata['video']);np.testing.assert_array_equal(w['start'],metadata['start'])
            terms={size:{k:np.asarray(v) for k,v in row['coefficients'][size].items()} for size in ['2049','4097']}
            bounds=interval_grid_difference(terms['2049'],terms['4097'])
            assert bounds['cost'].max()<=1e-4 and bounds['gradient'].max()<=1e-5
            for k,a in metadata['alpha'].items():np.testing.assert_array_equal(value_gradient(terms['4097'],a)[0],row['costs'][k])
            costs={k:value_gradient(terms['4097'],a)[0].tolist() for k,a in alpha.items()}
            costs.update(zero=row['costs']['old_zero'],prior_motion=row['costs']['old_contextual'])
            runs.append(dict(region=region,seed=row['seed'],costs=costs,video=w['video'].tolist(),start=w['start'].tolist()))
        values={k:np.array([r['costs'][k] for r in runs if r['region']==region]) for k in costs}
        out=dict(zero=float(values['zero'].mean()))
        for k,v in values.items():
            if k=='zero':continue
            delta=v-values['zero'];means=delta.mean((1,2))
            out[k]=dict(score=float(v.mean()),delta=float(delta.mean()),seed_deltas=means.tolist(),better_seeds=int((means<0).sum()),
                        horizon_delta=delta.mean((0,1)).tolist(),
                        per_video_delta={str(v):float(delta[:,w['video']==v].mean()) for v in np.unique(w['video'])})
        summary[region]=out
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,
                note='Frozen all5 group-risk candidates plus prior-motion and zero;cached631017-20/P64 holdprefix/tail24. '
                     'All original9scores exact replay,allalpha precision rechecked. NO new independent evidence/fit/rollout/integral/DEV/TEST/promotion.')
    (root/'group_risk_holdout_screen.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
