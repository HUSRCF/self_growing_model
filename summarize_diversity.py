"""Aggregate predeclared DEV particle-search comparisons without test tuning."""
import json
import argparse
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results')
    original=json.loads((root/'common300_original.json').read_text())
    reference=np.load(root/'common300_fixed2.npz')
    rows={};details={}
    for temperature in (.5,1.):
        label=f't{temperature:g}'
        files=[root/f'diversity_t{temperature:g}_s{s}.json' for s in (1729,2718,3141)]
        runs=[json.loads(p.read_text()) for p in files]
        details[label]=[]
        for p,run in zip(files,runs):
            a=np.load(p.with_suffix('.npz'))
            for key in ['history','truth','video','window_start']:
                np.testing.assert_array_equal(a[key],reference[key],err_msg=str(p)+key)
            per_video={}
            for v in np.unique(a['video']):
                mask=a['video']==v
                per_video[str(v)]=dict(rmse300=float(np.sqrt(a['per_window_embedding_mse'][mask,-1].mean())),
                                      energy300=float(a['per_window_energy'][mask,-1].mean()),
                                      failed300=float(a['failed'][mask,:,-1].mean()))
            unique=[len(np.unique(a['prediction'][i,:,-1],axis=0)) for i in range(len(a['video']))]
            details[label].append(dict(seed=run['config']['seed'],seconds=run['seconds'],
                                      rollbacks=run['rollbacks'],per_video=per_video,
                                      mean_unique_endpoints=float(np.mean(unique))))
        rows[label]={h:{metric:float(np.mean([r['score'][h][metric] for r in runs]))
                        for metric in ['embedding_rmse','energy_score','coverage90','failure']}
                     for h in ['50','100','300']}
    for particles in (1,8):
        runs=[r for r in original['results'] if r['particles']==particles]
        rows[f'original_p{particles}']={h:{metric:float(np.mean([r['score'][h][metric] for r in runs]))
                                           for metric in ['embedding_rmse','energy_score','coverage90','failure']}
                                       for h in ['50','100','300']}
    report=dict(results=rows,details=details,
                note='Means of per-seed metrics; 4 DEV videos,16 correlated windows. Search uses4 workers per run; times are not equal-resource speed comparisons. All outputs are offline rollouts.')
    (root/'diversity_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(rows,indent=2))


def expanded():
    root=Path('adaptive_search_results')
    original=json.loads((root/'diversity32_original_per_video.json').read_text())
    runs=[]; per_video=[]
    for seed in (1729,2718,3141):
        p=root/f'diversity32_t1_s{seed}.json'
        run=json.loads(p.read_text());runs.append(run)
        a=np.load(p.with_suffix('.npz'))
        np.testing.assert_array_equal(a['video'],original['video'])
        np.testing.assert_array_equal(a['window_start'],original['window_start'])
        scores={}
        from v20_rnn_mixture.engine.evaluate import metrics
        for v in np.unique(a['video']):
            m=a['video']==v
            scores[str(v)],_=metrics(a['prediction'][m],a['truth'][m],a['failed'][m])
        per_video.append(scores)
    def aggregate(rs):
        return {h:{k:float(np.mean([r['score'][h][k] for r in rs]))
                   for k in ['embedding_rmse','energy_score','coverage90','failure']}
                for h in ['50','100','300']}
    report=dict(search=aggregate(runs),original=aggregate(original['results']),per_video={})
    for v in sorted(set(map(str,original['video']))):
        report['per_video'][v]={h:dict(
            search_rmse=float(np.mean([r[v][h]['embedding_rmse'] for r in per_video])),
            original_rmse=float(np.mean([r['per_video'][v][h]['embedding_rmse'] for r in original['results']])))
            for h in ['50','100','300']}
    report['note']='Expanded DEV windows overlap the pilot videos/windows; not independent holdout. Three seeds,8 particles. Configuration t1 selected in pilot and frozen for this expansion.'
    report['search_seconds']=[r['seconds'] for r in runs]
    report['original_seconds']=[r['seconds'] for r in original['results']]
    report['rollbacks']=[r['rollbacks'] for r in runs]
    (root/'diversity32_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expanded',action='store_true')
    if ap.parse_args().expanded:expanded()
    else:main()
