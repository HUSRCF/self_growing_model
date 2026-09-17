"""Paired alternating CPU timing; exact output and search-counter parity."""
import json
from pathlib import Path
import numpy as np
from evaluate_adapted_checked import sparse_window
from v20_rnn_mixture.engine.data import tail_windows


def main():
    windows=tail_windows('dev',300,8);rows=[]
    # Fixed one window per DEV video, original policy; not chosen by speed.
    for repeat in range(2):
        for j,i in enumerate([0,8,16,24]):
            modes=['probe_cache','probe_fast']
            if (repeat+j)%2:modes.reverse()
            outputs={m:sparse_window((i,windows['history'][i],1729,None,m)) for m in modes}
            old,new=outputs['probe_cache'],outputs['probe_fast']
            for k in range(4):np.testing.assert_array_equal(old[k],new[k])
            assert old[4]==new[4]
            for k in old[5]:
                if k!='cpu_seconds':assert old[5][k]==new[5][k],k
            row=dict(repeat=repeat,window=i,order=modes,
                     cache_cpu=old[5]['cpu_seconds'],fast_cpu=new[5]['cpu_seconds'])
            rows.append(row);print(row,flush=True)
    old=sum(r['cache_cpu'] for r in rows);new=sum(r['fast_cpu'] for r in rows)
    report=dict(rows=rows,outputs_and_search_counters_exact=True,cache_cpu_sum=old,
                fast_cpu_sum=new,reduction_fraction=1-new/old,
                note='Two alternating repeats of four fixed DEV windows, original policy, seed1729,8particles. CPU sums include unchanged independent checker verification. Paired speed diagnostic, not independent accuracy replication.')
    Path('adaptive_search_results/fast_probe_paired_cpu.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
