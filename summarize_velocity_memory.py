"""Summarize saved trajectories, including U scores and shared-writer invariance."""
import json
from pathlib import Path
import numpy as np
from evaluate_protected_residual import distribution_scores


def main():
    root=Path('adaptive_search_results');source=json.loads((root/'prefix_velocity_memory_pilot.json').read_text())
    summary={};shared_reference=None
    for name,rows in source['runs'].items():
        distributions=[]
        for r in rows:
            with np.load(root/f'prefix_velocity_memory_{name}_{r["seed"]}.npz') as z:
                distributions.append(distribution_scores(z['prediction'],z['truth'],z['failed'],z['video']))
                if name=='memory_shared':
                    p=z['prediction']
                    np.testing.assert_array_equal(p,np.repeat(p[:,:1],p.shape[1],axis=1))
                    if shared_reference is None:shared_reference=p.copy()
                    np.testing.assert_array_equal(p,shared_reference)
        summary[name]=dict(objective=source['mean_objective'][name],
            rmse={str(t):float(np.mean([r['score'][str(t)]['embedding_rmse'] for r in rows])) for t in [50,100,300]},
            energy_u={str(t):float(np.mean([d[str(t)]['mean'] for d in distributions])) for t in [50,100,300]},
            energy_v={str(t):float(np.mean([r['score'][str(t)]['energy_score'] for r in rows])) for t in [50,100,300]},
            max_numeric_failure=max(r['score']['300']['failure'] for r in rows),
            per_video_rmse3={str(v):float(np.mean([r['per_video'][str(v)]['300']['embedding_rmse'] for r in rows])) for v in [18,16,13]})
    result=dict(summary=summary,shared_particle_and_action_seed_invariance_exact=True,
                note='Scoring existing saved TRAINhold trajectories only. Sharedlocal0 has exactly identical numerical trajectories acrossparticles/actionseeds; q/event still sampled but cannot change this writer output. No new parameter selection/DEV/TEST.')
    (root/'prefix_velocity_memory_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
