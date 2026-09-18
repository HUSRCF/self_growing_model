import unittest
import numpy as np
from validate_delayed_root_regions import summarize


class DelayedRootRegionsTests(unittest.TestCase):
    def test_paired_horizon_summary_and_short_invariant(self):
        base=np.ones((2,3));delta=np.array([[0.,-.1,.2],[0.,-.3,-.2]])
        rows=[dict(baseline=base.tolist(),mixed=(base+scale*delta).tolist()) for scale in [1.,2.]]
        r=summarize(rows,np.array([13,18]))
        np.testing.assert_allclose(r['horizon_delta'],[0.,-.3,0.],atol=1e-14)
        self.assertAlmostEqual(r['delta'],-.1)
        rows[0]['mixed'][0][0]+=.001
        with self.assertRaises(AssertionError):summarize(rows,np.array([13,18]))
