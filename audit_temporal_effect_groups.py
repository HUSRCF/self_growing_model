"""Post-hoc descriptive motion groups; reuse all saved scores, no rollouts."""
import json
from pathlib import Path
import numpy as np
from audit_writer_temporal_shift import motion
from v20_rnn_mixture.engine.data import load_video


def main():
    root=Path('adaptive_search_results');tail=json.loads((root/'writer_temporal_shift_audit.json').read_text())
    prefix=json.loads((root/'truncated_writer_confirmation.json').read_text());edges=tail['fit_prefix_motion_edges'];out={}
    for region,r in [('prefix',prefix),('tail',tail)]:
        meta=tail['states'][region]
        h=np.stack([load_video(v)[t-31:t+1] for v,t in zip(meta['video'],meta['window_start'])])
        bins=np.searchsorted(edges,motion(h),side='right');out[region]={}
        base=np.array([x['window_u_cost'] for x in r['runs']['zero']])
        for name,rows in r['runs'].items():
            delta=np.array([x['window_u_cost'] for x in rows])-base;groups=[]
            for k in range(4):
                mask=bins==k;values=delta[:,mask].mean(1) if mask.any() else None
                groups.append(dict(bin=k,windows=int(mask.sum()),window_indices=np.flatnonzero(mask).tolist(),
                                   mean_delta=None if values is None else float(values.mean()),
                                   conditional_seed_se=None if values is None else float(values.std(ddof=1)/np.sqrt(len(values)))))
            out[region][name]=groups
    report=dict(edges=edges,groups=out,note='Post-hoc description using preexisting fit-prefix thresholds,not new model selection or causal evidence. All bins/windows retained,small high-motion tail bins explicitly counted. No refitting/rollout/DEV/TEST; do not infer universal speed-gating rule.')
    (root/'temporal_effect_groups.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
