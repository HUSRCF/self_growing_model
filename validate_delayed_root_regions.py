"""Fixed delayed intervention across temporal and adapter-video holdouts."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from confirm_delayed_root import evaluate
from validate_temporal_residual_gate import windows
from confirm_temporal_kernel_holdout import hold_windows


def summarize(rows,video):
    d=np.asarray([np.asarray(r['mixed'])-r['baseline'] for r in rows]);seed=d.mean((1,2));hs=d.mean(1)
    assert np.count_nonzero(d[:,:,0])==0
    return dict(baseline=float(np.mean([r['baseline'] for r in rows])),delta=float(d.mean()),
                seed_delta=seed.tolist(),conditional_seed_se=float(seed.std(ddof=1)/np.sqrt(len(seed))),
                horizon_delta=d.mean((0,1)).tolist(),seed_horizon_delta=hs.tolist(),
                per_video_horizon={str(v):d[:,video==v].mean((0,1)).tolist() for v in np.unique(video)})


def worker(args):
    region,w,seed=args
    return dict(region=region,**evaluate((w,seed,32)))


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'crossfit_delayed_root_model.json',directory/'delayed_root_confirmation.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model=json.loads(paths[0].read_text());old=json.loads(paths[1].read_text())
    for p,h in model['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert all(m['root']==6 and m['alpha']==.25 for m in model['folds'].values())
    fit=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for key in ['video','start']:np.testing.assert_array_equal(fit[key],model[key])
    replay=evaluate((fit,781017,32))
    for key in ['baseline','mixed','guards','short_roundoff']:
        np.testing.assert_array_equal(replay[key],old['rows'][0][key])
    print('First old confirmation stream exact; policy unchanged',flush=True)
    sets={'fit_late':windows('evaluation'),'hold_prefix':hold_windows('prefix'),'hold_tail':hold_windows('tail')}
    for v in np.unique(fit['video']):
        assert fit['start'][fit['video']==v].max()+300 < sets['fit_late']['start'][sets['fit_late']['video']==v].min()-31
    for region in ['hold_prefix','hold_tail']:assert not set(sets[region]['video'])&set(fit['video'])
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(worker,[(region,w,seed) for region,w in sets.items() for seed in range(791017,791021)]):
            rows.append(row)
            print(row['region'],row['seed'],np.mean(np.asarray(row['mixed'])-row['baseline'],axis=0),'guards',sum(row['guards']),flush=True)
    summary={region:summarize([r for r in rows if r['region']==region],w['video']) for region,w in sets.items()}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,rows=rows,metadata={region:{k:w[k].tolist() for k in ['video','start']} for region,w in sets.items()},source_hashes=hashes,
                note='Frozen delayedr6/.25 at51stpoint. Fit-late40 temporal history/targets isolated; adapterhold18/16/13 prefix24/tail24, no fitting videos overlap. Fixed791017-20/P32 two components,first50 paths/failures exact. All3regions evaluated regardless of intermediate results. Historical states reused and backboneTRAIN,NOT projectblind. Tailfirsthistory may cross midpoint; targetsecondhalf,withinregion overlap. ConditionalRNG SE notvideoSE. No tuning/timing sweep/DEV/TEST/promotion.')
    (directory/'delayed_root_regions.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
