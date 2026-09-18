import unittest
import numpy as np
from audit_q_prior_support import describe


class PriorSupportTests(unittest.TestCase):
    def test_bins_and_no_mutation(self):
        p=np.array([0.,.01,.1,.5,1.]);y=np.tile(np.arange(5),(4,1)).astype(float);pred=-np.ones(5)
        old=y.copy();r=describe(p,y,pred,np.array([1,1,1,2,2]))
        self.assertEqual([b['n'] for b in r['bins']],[1,1,1,2])
        self.assertEqual(r['actual_delta'],2.);self.assertEqual(r['optimism_gap'],3.)
        np.testing.assert_array_equal(y,old)
        empty=describe(np.array([1.,1.]),np.zeros((4,2)),None,np.array([1,2]))
        self.assertNotIn('actual_delta',empty['bins'][0])
        self.assertIsNone(empty['logprior_vs_delta'])


if __name__=='__main__':unittest.main()
