"""Paired width-only comparison, reusing frozen gates and exact prior RNG streams."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from validate_temporal_residual_gate import windows,gather,passes,new_weights
from validate_soft_gates_new_windows import causal_features
from validate_full_soft_gates import weights
from soft_output_mixture import value_gradient


def main():
    root=Path('adaptive_search_results')
    paths=[root/'temporal_kernel_model.json',root/'temporal_kernel_training_audit.json',
           root/'temporal_residual_model.json',root/'full_soft_gate_model.json',root/'temporal_residual_evaluation.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fitted=json.loads(paths[0].read_text());audit=json.loads(paths[1].read_text())
    assert fitted['all_converged'] and audit['all_precision_gates_pass']
    assert audit['source_hashes'][str(paths[0])]==hashes[str(paths[0])]
    for p,h in audit['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    kernel=fitted['model'];new=json.loads(paths[2].read_text())['model'];old=json.loads(paths[3].read_text())['model']
    previous=json.loads(paths[4].read_text());late=windows('evaluation')
    np.testing.assert_array_equal(late['video'],previous['video']);np.testing.assert_array_equal(late['start'],previous['start'])
    motion,raw=causal_features(AdaptiveBeam(),late)
    alpha={**{'old_'+k:v for k,v in weights(old,motion,raw).items()},**new_weights(new,motion,raw)}
    for k,a in alpha.items():np.testing.assert_array_equal(a,previous['alpha'][k])
    runs=[]
    for before in previous['runs']:
        row=gather(kernel,motion,late,before['seed'])
        fine={k:np.asarray(v) for k,v in row['coefficients']['4097'].items()}
        np.testing.assert_array_equal(fine['baseline'],before['coefficients']['4097']['baseline'])
        row.update(seed=before['seed'],costs={k:value_gradient(fine,a)[0].tolist() for k,a in alpha.items()})
        runs.append(row);print('paired evaluation',before['seed'],passes(row),flush=True)
    summary={}
    baseline=np.asarray([r['costs']['old_zero'] for r in previous['runs']])
    for k in alpha:
        value=np.asarray([r['costs'][k] for r in runs]);prior=np.asarray([r['costs'][k] for r in previous['runs']])
        delta=value-baseline;change=value-prior
        summary[k]=dict(score=float(value.mean()),delta=float(delta.mean()),width_change=float(change.mean()),
                        seed_deltas=delta.mean((1,2)).tolist(),seed_width_changes=change.mean((1,2)).tolist(),
                        horizon_delta=delta.mean((0,1)).tolist(),
                        per_video_delta={str(v):float(delta[:,late['video']==v].mean()) for v in np.unique(late['video'])})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,runs=runs,source_hashes=hashes,all_precision_gates_pass=all(passes(r) for r in runs),
                note='Only width changed. Same frozen9alpha,late40windows,622017/18 P32 as previous pilot;baseline/alpha exact replay. '
                     'New TRAIN-only candidate on historically used internal validation,NOT independent confirmation. '
                     'No gate refit,threshold change,hold/DEV/TEST/promotion. All strategies retained.')
    (root/'temporal_kernel_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
