"""Fixed reference-path hold diagnostics; never used for fitting or selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from multistep_initializer_pilot import ResidualInitializer,observed_routes,fixed_route_residual
from velocity_memory_pilot import load_block
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root=Path('adaptive_search_results');paths=[root/'prefix_velocity_memory_model.npz']+[root/f'multistep_initializer_h{h}.npz' for h in [1,10]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    s=AdaptiveBeam();_,Writer=legacy_types();weak=Writer(LegacyFeatureBridge(s.base),load_block(paths[0]),dict(learned_initialization=True))
    models={'base':weak}
    for h,path in zip([1,10],paths[1:]):
        with np.load(path) as z:models[f'h{h}']=ResidualInitializer(weak,z['coef'].copy(),z['scale'].copy())
    # Same starts as actual300-step evaluation, only first10 targets used here.
    w=prefix_windows(SPLITS['train'][-3:],steps=300,per_video=8)
    w=dict(w,truth=w['truth'][:,:10]);q,r=observed_routes(s.base,w);report={}
    for name,model in models.items():
        e=fixed_route_residual(weak,w,q,r,model.initialize(w['history']),10)
        report[name]={str(h):dict(mse=float(np.mean(e[:,:h]**2)),
            per_video={str(v):float(np.mean(e[w['video']==v,:h]**2)) for v in np.unique(w['video'])}) for h in [1,10]}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    out=dict(results=report,source_hashes=hashes,sources_unchanged=True,
             note='Post-fit fixed observed-q-path TRAINhold diagnostics on same24starts as300step evaluation. Uses future observed labels/path, not deployable inference; no refit/selection/DEV/TEST.')
    (root/'multistep_initializer_hold_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
