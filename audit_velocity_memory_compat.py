"""No-fit TRAIN-prefix compatibility and event-role audit of legacy writer."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import prefix_windows
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def event_mean_variance(pe,tr,values):
    means=np.einsum('ber,brd->bed',tr,values)
    overall=np.einsum('be,bed->bd',pe,means)
    return (pe*((means-overall[:,None])**2).mean(-1)).sum(1)


def main():
    root=Path('adaptive_search_results');legacyroot=Path('v20_complete')
    paths=[legacyroot/'reference/q_old_mlp_poly_blend0.5.json',
           Path('v20_rnn_mixture/models/frozen_dynamics.json'),
           legacyroot/'v19_models/old_weak_L8_o3_initialized_block.json',
           legacyroot/'v19_models/old_weak_learned_init_L8.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old,current,block,config=[json.loads(p.read_text()) for p in paths]
    equality={key:old.get(key)==value for key,value in current.items()};assert all(equality.values())
    Old,Writer=legacy_types();oldbase=Old(old);s=AdaptiveBeam();bridge=LegacyFeatureBridge(s.base)
    legacy=Writer(oldbase,block,config['repair']);adapted=Writer(bridge,block,config['repair'])
    shared=Writer(bridge,block,dict(config['repair'],local_fraction=0.))
    w=prefix_windows(SPLITS['train'],steps=1,per_video=16);h=w['history'];q,mem=s.machine.initialize(h)
    pe,tr,_=s.machine.read(h,q,mem);pe0=pe.copy();tr0=tr.copy()
    np.testing.assert_array_equal(q,oldbase.state_from_history(h)[0])
    global0,z0=oldbase.neural_values(h);global1,z1=bridge.neural_values(h)
    np.testing.assert_allclose(global0,global1,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(z0,z1,rtol=1e-13,atol=1e-14)
    v0=legacy.initialize(h);v=adapted.initialize(h);np.testing.assert_allclose(v0,v,rtol=1e-13,atol=1e-14)
    hh=np.repeat(h,8,axis=0);qq=np.repeat(q,8);rr=np.tile(np.arange(8),len(h));vv=np.repeat(v,8,axis=0)
    original_f=oldbase.execute_rule(hh,qq,rr);current_f=s.base.execute_rule(hh,qq,rr)
    np.testing.assert_allclose(original_f,current_f,rtol=1e-13,atol=1e-14)
    y0,nv0=legacy.execute(hh,qq,rr,vv);y,nv=adapted.execute(hh,qq,rr,vv)
    np.testing.assert_allclose(y0,y,rtol=1e-13,atol=1e-14);np.testing.assert_allclose(nv0,nv,rtol=1e-13,atol=1e-14)
    sy,sv=shared.execute(hh,qq,rr,vv);groups={}
    for name,output,next_v in [('current',current_f,None),('legacy_local',y,nv),('legacy_shared',sy,sv)]:
        output=output.reshape(len(h),8,2)
        groups[name]=dict(mean_destination_span_rad=float(np.ptp(output,axis=1).max(1).mean()),
            nonidentical_destination_histories=int((np.ptp(output,axis=1).max(1)>1e-12).sum()),
            event_conditional_mean_angle_variance=float(event_mean_variance(pe,tr,output).mean()))
        if next_v is not None:groups[name]['event_conditional_mean_velocity_variance']=float(event_mean_variance(pe,tr,next_v.reshape(len(h),8,2)).mean())
    np.testing.assert_array_equal(sy.reshape(len(h),8,2),np.repeat(sy.reshape(len(h),8,2)[:,:1],8,axis=1))
    pe1,tr1,_=s.machine.read(h,q,mem);np.testing.assert_array_equal(pe0,pe1);np.testing.assert_array_equal(tr0,tr1)
    lengths=[len(load_video(v)) for v in SPLITS['train']]
    counts=dict(stored_weak_rows=block['training']['n'],full_train_weak_rows=sum(n-44 for n in lengths),
                all_train_prefix_weak_rows=sum(n//2-44 for n in lengths),
                fit10_prefix_weak_rows=sum(n//2-44 for n in lengths[:-3]),
                stored_initializer_rows=block['initializer']['n'],full_train_initializer_rows=sum(n-40 for n in lengths))
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    report=dict(base_field_equality=equality,source_sha256=hashes,sources_unchanged=True,
        histories=len(h),initializer_feature_dimension=z1.shape[1]+8,legacy_initializer_dimension=len(block['initializer']['scale']),
        maximum_initializer_difference=float(np.abs(v-v0).max()),maximum_writer_difference=float(np.abs(y-y0).max()),
        maximum_next_velocity_difference=float(np.abs(nv-nv0).max()),event_read_unchanged=True,
        destination_effect=groups,training_scope=counts,units=block['training'],
        note='Compatibility audit only, no fitting/predictive DEV/TEST evaluation. Histories all TRAIN prefixes; fullTRAIN lengths used only for lineage counts. Legacy saved sample counts match all13fullTRAIN, not current10fitprefix protocol. Existing old-q partition/base identical; old class API needs bridge. Current GRU read untouched; event-conditional mean variation is not full mutual information/long-horizon necessity proof. Shared local_fraction0 removes immediate destination dependence. Do not deploy legacy weights as newly validated prefix-trained model.')
    (root/'velocity_memory_compatibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
