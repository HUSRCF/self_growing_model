import unittest
import numpy as np
from initial_velocity_distribution import InitialVelocityDistribution, residual_covariance


class DistributionTests(unittest.TestCase):
    def test_video_weighting(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = np.array([[5., -1.]])
        m, c = residual_covariance([a,b])
        m2, c2 = residual_covariance([np.tile(a,(7,1)),b])
        np.testing.assert_allclose(m, m2); np.testing.assert_allclose(c, c2)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(), -1e-12)

    def test_sampling_and_zero(self):
        class Weak:
            def initialize(self, h): return h[:,-1].copy()
            def execute(self, h,q,r,v): return h[:,-1]+v, v
        h = np.zeros((100000,1,2)); weak = Weak()
        z = InitialVelocityDistribution(weak, np.zeros((2,2)), 8)
        np.testing.assert_array_equal(z.initialize(h), weak.initialize(h))
        cov = np.array([[2.,.4],[.4,1.]])
        model = InitialVelocityDistribution(weak, cov, 8)
        v = model.initialize(h)
        np.testing.assert_array_equal(v, model.initialize(h))
        np.testing.assert_allclose(np.cov(v.T), cov, atol=.025)
        a,b=model.execute(h,None,None,v)
        np.testing.assert_array_equal(a,v);np.testing.assert_array_equal(b,v)
        with self.assertRaises(ValueError): InitialVelocityDistribution(weak,-cov,8)


if __name__ == '__main__': unittest.main()
