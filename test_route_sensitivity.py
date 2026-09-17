import unittest
import numpy as np
from audit_route_sensitivity import interval_overlap,shared_uniform_mismatch


class SensitivityTests(unittest.TestCase):
    def test_exact_coupling(self):
        p=np.array([[.8,.2]]);q=np.array([[.2,.8]])
        overlap=interval_overlap(p,q)
        np.testing.assert_allclose(overlap.sum(2),p);np.testing.assert_allclose(overlap.sum(1),q)
        T=np.array([[[1.,0.],[0.,1.]]])
        e,r=shared_uniform_mismatch(p,T,q,T)
        np.testing.assert_allclose(e,.6);np.testing.assert_allclose(r,.6)
        same=np.array([[[.3,.7],[.3,.7]]])
        _,r=shared_uniform_mismatch(p,same,q,same)
        np.testing.assert_allclose(r,0,atol=1e-15)
        e,r=shared_uniform_mismatch(p,T,p,T)
        np.testing.assert_allclose(e,0,atol=1e-15);np.testing.assert_allclose(r,0,atol=1e-15)


if __name__=='__main__':unittest.main()
