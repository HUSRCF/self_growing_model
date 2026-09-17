import unittest
import numpy as np
from writer_compatible_initializer import solve_velocity


class InitializerTests(unittest.TestCase):
    def test_linear_inverse_and_bound(self):
        initial=np.zeros((3,2));delta=np.full_like(initial,.01)
        v,residual=solve_velocity(delta,initial,lambda x:.1*x)
        np.testing.assert_allclose(v,delta/1.05,atol=1e-12)
        np.testing.assert_allclose(residual,0,atol=1e-12)
        v,residual=solve_velocity(np.ones_like(initial),initial,lambda x:np.zeros_like(x))
        np.testing.assert_allclose(v,.02);np.testing.assert_allclose(residual,-.98)
        np.testing.assert_array_equal(initial,np.zeros_like(initial))


if __name__=='__main__':unittest.main()
