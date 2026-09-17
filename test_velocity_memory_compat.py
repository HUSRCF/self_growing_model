import unittest
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from velocity_memory_compat import LegacyFeatureBridge,legacy_types
from audit_velocity_memory_compat import event_mean_variance
from train_closed_loop_policy import prefix_windows


class VelocityMemoryCompatibilityTests(unittest.TestCase):
    def test_feature_bridge_and_initialization_match_legacy(self):
        Old,Writer=legacy_types();base=AdaptiveBeam().base
        old=Old(json.loads(Path('v20_complete/reference/q_old_mlp_poly_blend0.5.json').read_text()))
        block=json.loads(Path('v20_complete/v19_models/old_weak_L8_o3_initialized_block.json').read_text())
        h=prefix_windows([12],steps=1,per_video=2)['history'];bridge=LegacyFeatureBridge(base)
        for a,b in zip(old.neural_values(h),bridge.neural_values(h)):np.testing.assert_allclose(a,b,atol=1e-14)
        a=Writer(old,block,dict(learned_initialization=True)).initialize(h)
        b=Writer(bridge,block,dict(learned_initialization=True)).initialize(h)
        np.testing.assert_allclose(a,b,atol=1e-14)

    def test_event_mean_variance_tracks_destination_dependence(self):
        pe=np.array([[.5,.5]]);tr=np.array([[[1.,0],[0,1.]]])
        values=np.array([[[0.,0],[2.,2]]])
        np.testing.assert_allclose(event_mean_variance(pe,tr,values),1)
        np.testing.assert_allclose(event_mean_variance(pe,tr,np.ones_like(values)),0)


if __name__=='__main__':unittest.main()
