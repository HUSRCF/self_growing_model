import unittest
from unittest.mock import patch
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from audit_residual_direction import state_snapshots
from drift_residual_pilot import generated_rows
from train_closed_loop_policy import prefix_windows


class DriftResidualTests(unittest.TestCase):
    def test_generated_features_ignore_targets_and_match_frozen_rollout(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=64,per_video=16)
        # Only the first history block initializes the free carrier.
        observed=dict(history=np.tile(w['history'],(64,1,1)),
                      q=np.zeros(1024,dtype=int),truth=w['truth'].transpose(1,0,2).reshape(-1,2),
                      probability=np.full((1024,8),1/8),video=np.full(1024,12))
        changed={k:v.copy() for k,v in observed.items()}
        changed['truth']+=1.;changed['history'][16:]+=2.
        with patch('drift_residual_pilot.SPLITS',{'train':[12]}):
            actual=generated_rows(s,observed)
            other=generated_rows(s,changed)
        for key in ['history','q','probability','video']:
            np.testing.assert_array_equal(actual[key],other[key])
        np.testing.assert_array_equal(actual['truth'],observed['truth'])
        snapshots=state_snapshots(s,w['history'],41017+12,steps=24)
        for t,(h,q,hidden,dead) in snapshots.items():
            np.testing.assert_array_equal(actual['history'][16*t:16*(t+1)],h)
            np.testing.assert_array_equal(actual['q'][16*t:16*(t+1)],q)
            self.assertFalse(dead.any())
        np.testing.assert_array_equal(observed['history'][:16],w['history'])


if __name__=='__main__':unittest.main()
