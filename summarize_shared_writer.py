"""Numerical parity, CPU cost and finite revision-window diagnostics."""
import json
from pathlib import Path
import numpy as np


def compare(left,right,strict=True):
    a=np.load(left);b=np.load(right)
    for k in ['history','truth','video','window_start']+(['failed'] if strict else []):
        np.testing.assert_array_equal(a[k],b[k],err_msg=k)
    diff=float(np.max(np.abs(a['prediction']-b['prediction'])))
    if strict:np.testing.assert_allclose(a['prediction'],b['prediction'],rtol=1e-10,atol=1e-9)
    return diff


def main():
    root=Path('adaptive_search_results'); report={}
    for mode in ['full','sparse']:
        a=json.loads((root/f'shared32_ref_{mode}.json').read_text())
        b=json.loads((root/f'shared32_shared_{mode}.json').read_text())
        difference=compare(root/f'shared32_ref_{mode}.npz',root/f'shared32_shared_{mode}.npz')
        for k in ['rollbacks','revision']:assert a[k]==b[k]
        for k in a['audit']:
            if k!='worker_cpu_seconds':assert a['audit'][k]==b['audit'][k],k
        report[mode]=dict(max_prediction_abs_difference=difference,
                          reference_cpu=a['audit']['worker_cpu_seconds'],shared_cpu=b['audit']['worker_cpu_seconds'],
                          cpu_speedup=a['audit']['worker_cpu_seconds']/b['audit']['worker_cpu_seconds'],
                          revision=b['revision'],rollbacks=b['rollbacks'])
    buffer=[]
    for seed in [1729,2718,3141]:
        a=json.loads((root/f'buffer2_sparse_s{seed}.json').read_text())
        diff=compare(root/f'buffer2_sparse_s{seed}.npz',root/f'sparse32_period5_matched_s{seed}.npz',strict=False)
        assert a['revision']['max_rollback_distance']<=2
        buffer.append(dict(seed=seed,max_prediction_abs_difference=diff,revision=a['revision'],
                           failure=a['score']['300']['failure'],cpu=a['audit']['worker_cpu_seconds'],
                           rmse300=a['score']['300']['embedding_rmse']))
    report['buffer2']=buffer
    p=root/'shared32_sparse_s3141_unbounded.json'
    if p.exists():
        run=json.loads(p.read_text())
        diff=compare(p.with_suffix('.npz'),root/'sparse32_period5_matched_s3141.npz')
        report['unbounded3141']=dict(revision=run['revision'],failure=run['score']['300']['failure'],
                                    max_prediction_abs_difference=diff,rollbacks=run['rollbacks'])
    report['note']='32DEV windows,8particles. Finite revision distance is NOT measured real-time latency; the prototype still returns a whole offline trajectory. Shared matrix operations allow small roundoff.'
    (root/'shared_writer_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
