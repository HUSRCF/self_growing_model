import unittest
import numpy as np
from multistep_initializer_pilot import bounded_gauss_newton


class MultistepTests(unittest.TestCase):
    def test_recovery_monotonicity_and_bound(self):
        initial=np.zeros((3,2));target=np.array([[.01,-.01],[.005,.008],[1.,-1.]])
        def residual(v):return (v-target)[:,None,:]*np.arange(1,11)[None,:,None]
        v,trace=bounded_gauss_newton(initial,residual)
        np.testing.assert_allclose(v,np.clip(target,-.02,.02),atol=1e-10)
        self.assertTrue(np.all(np.diff(trace)<=1e-14))
        np.testing.assert_array_equal(initial,np.zeros_like(initial))

    def test_zero_objective_stays_zero(self):
        initial=np.ones((2,2))*.1
        v,trace=bounded_gauss_newton(initial,lambda x:np.zeros((2,3,2)))
        np.testing.assert_array_equal(v,initial);self.assertEqual(trace,[0.]*6)


if __name__=='__main__':unittest.main()
