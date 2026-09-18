import json
import unittest
import numpy as np
from refit_matched_kernel_gates import fit_model
from validate_temporal_residual_gate import new_weights


class MatchedKernelGateTests(unittest.TestCase):
    def test_fixed_training_fit_roundtrip_and_bounds(self):
        rng=np.random.default_rng(79)
        raw=rng.normal(size=(24,61));motion=np.exp(rng.normal(-2,.3,24))
        terms=dict(linear=rng.normal(-.01,.01,(24,3)),quadratic=rng.normal(.02,.005,(24,3)))
        model=fit_model(motion,raw,terms)
        first=new_weights(model,motion,raw);second=new_weights(json.loads(json.dumps(model)),motion,raw)
        for k,a in first.items():
            self.assertEqual(a.shape,(24,3))
            self.assertTrue(np.isfinite(a).all() and ((a>=0)&(a<=1)).all())
            np.testing.assert_array_equal(a,second[k])
        self.assertTrue((np.abs(first['residual']-first['new_motion'])<=.25+1e-15).all())
