import unittest
import numpy as np
from audit_grouped_root_feasibility import grouped_moments,analyze


class GroupedRootFeasibilityTests(unittest.TestCase):
    def test_inactive_windows_denominator_and_uniform_embedding(self):
        l=np.ones((4,2,3));q=np.broadcast_to(np.eye(2),(4,3,2,2)).copy()
        group=np.array([0,1,1,1]);video=np.ones(4,int)
        gl,gq=grouped_moments(l,q,group,video)
        np.testing.assert_array_equal(gl[0,:,0],[.25,.25,.75,.75])
        s=np.array([.2,.8]);z=np.tile(.25*s,2)
        expected=.25+.25**2*(s@s)
        np.testing.assert_allclose(np.einsum('vrh,r->vh',gl,z)+np.einsum('r,vhrs,s->vh',z,gq,z),expected)

    def test_fit_only_partition_and_strength_cap(self):
        l=-np.ones((4,2,3));q=np.zeros((4,3,2,2));motion=np.array([1.,2.,3.,4.]);v=np.array([1,1,2,2])
        result=analyze(l,q,motion,v)
        self.assertEqual(result['threshold'],2.5)
        self.assertEqual(result['counts'],[2,2])
        self.assertTrue(np.all(np.array(result['bin_actual_strength'])<=.25+1e-12))
        self.assertLess(result['direction_fit']['mean_delta'],0.)
        # Both bins can attain .25, not an artificial shared total .25 limit.
        np.testing.assert_allclose(result['bin_actual_strength'],[.25,.25],atol=1e-12)
