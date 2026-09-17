"""One-window CPU profile; descriptive overhead-instrumented diagnostic."""
import cProfile
import json
import pstats
import time
from pathlib import Path
import numpy as np
from evaluate_adapted_checked import sparse_window
from v20_rnn_mixture.engine.data import tail_windows


def main():
    w=tail_windows('dev',300,8);pr=cProfile.Profile(timer=time.process_time)
    pr.enable();result=sparse_window((0,w['history'][0],1729,None,'probe_cache'));pr.disable()
    ref=np.load('adaptive_search_results/adapted_probe_model0_roll1729.npz')
    np.testing.assert_array_equal(result[0],ref['prediction'][0])
    np.testing.assert_array_equal(result[1],ref['failed'][0])
    stats=pstats.Stats(pr);rows=[]
    for (filename,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
        rows.append(dict(file=Path(filename).name,line=line,function=name,calls=nc,
                         self_cpu_seconds=tt,cumulative_cpu_seconds=ct))
    rows.sort(key=lambda x:x['self_cpu_seconds'],reverse=True)
    report=dict(window=0,model_seed=0,rollout_seed=1729,mode='probe_cache',prediction_bitwise_equal=True,
        total_profiled_cpu_seconds=stats.total_tt,top_self=rows[:25],
        selected=[r for r in rows if r['function'] in ['_expand','_read','read','step','log_density_all','execute',
                                                     '_execute_candidates','membership','score','reject','_value']],
        note='One selected DEV window under cProfile with process-time timer. Profiling overhead changes timing; not a benchmark or independent window sample. Cumulative times overlap and must not be added.')
    Path('adaptive_search_results/commit_probe_cpu_profile.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
