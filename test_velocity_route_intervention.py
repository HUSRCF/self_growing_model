import unittest
import numpy as np
from audit_velocity_route_intervention import replay_edges


class ReplayTests(unittest.TestCase):
    def test_edge_chain_and_fixed_writer(self):
        class W:
            def initialize(self,h):return np.ones_like(h[:,-1])*.01
            def execute(self,h,q,r,v):return h[:,-1]+v+r[:,None]*.001,v.copy()
        h=np.zeros((1,32,2));src=np.array([[0]*4,[1]*4]);dst=np.array([[1]*4,[2]*4])
        p,f=replay_edges(h,W(),src,dst,4)
        np.testing.assert_allclose(p[0,0],[[.011,.011],[.023,.023]])
        self.assertFalse(f.any())
        with self.assertRaises(AssertionError):replay_edges(h,W(),np.zeros_like(src),dst,4)

    def test_failure_freezes_state(self):
        class W:
            def initialize(self,h):return np.zeros_like(h[:,-1])
            def execute(self,h,q,r,v):return np.full_like(v,np.nan),v.copy()
        h=np.ones((1,32,2));src=np.array([[0]*4,[1]*4]);dst=src+1
        p,f=replay_edges(h,W(),src,dst,4)
        np.testing.assert_array_equal(p,np.ones_like(p));self.assertTrue(f.all())


if __name__=='__main__':unittest.main()
