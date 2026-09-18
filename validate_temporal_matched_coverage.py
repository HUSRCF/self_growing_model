"""Frozen matched-budget critics transferred to purged late TRAIN windows."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from evaluate_conditional_policy import evaluate,summarize
from validate_temporal_residual_gate import windows


def assert_purged(train,late):
    if set(train['video'])!=set(late['video']):raise ValueError('Video sets must match')
    for v in np.unique(train['video']):
        if train['start'][train['video']==v].max()+300 >= late['start'][late['video']==v].min()-31:
            raise ValueError('Fit targets overlap late histories')


def comparisons(rows,video):
    out={}
    for a,b in [('multi_context','single_context'),('multi_action','single_action'),('multi_context','fixed_r6')]:
        d=np.asarray([np.asarray(r['mixed'][a])-r['mixed'][b] for r in rows])
        out[a+'_minus_'+b]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),horizon_delta=d.mean((0,1)).tolist(),
                                per_video_horizon={str(v):d[:,video==v].mean((0,1)).tolist() for v in np.unique(video)})
    return out


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'matched_state_coverage_model.json',directory/'matched_state_coverage_evaluation.json']
    model,old=[json.loads(p.read_text()) for p in paths]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    train=TrainingPrefixPool(AdaptiveBeam().base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(train[k],model[k])
    previous=old['rows'][0];assert previous['seed']==831017
    replay=evaluate((train,model,831017,16))
    np.testing.assert_array_equal(replay['baseline'],previous['baseline'])
    assert replay['guards']==previous['guards'];assert replay['choice_counts']==previous['choice_counts']
    for name in replay['mixed']:
        np.testing.assert_array_equal(replay['mixed'][name],previous['mixed'][name])
        for k in ['linear','quadratic']:np.testing.assert_array_equal(replay['coefficients'][name][k],previous['coefficients'][name][k])
    print('All old831017 policies/guards/choices/LQ exactly replayed',flush=True)
    late=windows('evaluation');assert_purged(train,late)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,[(late,model,seed,16) for seed in range(841017,841021)]):
            rows.append(row);print('Frozen late transfer',row['seed'],'guards',row['guards'],flush=True)
    summary=summarize(rows,late['video'],('multi_context','single_context'))
    # Remove legacy field names: this run compares coverage, not context vs action.
    del summary['context_minus_action'];del summary['context_minus_action_seed']
    compare=comparisons(rows,late['video'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(summary=summary,comparisons=compare,rows=rows,video=late['video'].tolist(),start=late['start'].tolist(),source_hashes=hashes,
                note='Four matched-budget LOVO critics+fixedr6 allFROZEN/.25t50. Old831017allscore/LQ/guard/choice exact. Late40/same10TRAINvideos, allfit target endpoints before earliest latehistory, new841017-20/P16allparticles/fullfiniteU. First50exact, sourcehashsame, no retrain/tuning/late-driven modelchoice/hold/DEV/TEST/defaultpromotion. Historical late region reused,NOTblindtest; backboneTRAIN,critic-videoexcluded only.')
    (directory/'temporal_matched_coverage_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(summary=summary,comparisons=compare),indent=2),flush=True)


if __name__=='__main__':main()
