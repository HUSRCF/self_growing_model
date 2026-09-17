import unittest
import numpy as np
from conditional_velocity_distribution import ConditionalVelocityDistribution, grouped_covariances
from initial_velocity_distribution import InitialVelocityDistribution


class ConditionalTests(unittest.TestCase):
    def test_equal_covariance_and_zero(self):
        class Writer:
            def initialize(self,h):return h[:,-1].copy()
        w=Writer();h=np.random.default_rng(4).normal(size=(100,32,2));cov=np.array([[2.,.2],[.2,1.]])
        model=ConditionalVelocityDistribution(w,[1,2,3],np.repeat(cov[None],4,axis=0),9)
        np.testing.assert_array_equal(model.initialize(h),InitialVelocityDistribution(w,cov,9).initialize(h))
        zero=ConditionalVelocityDistribution(w,[1,2,3],np.zeros((4,2,2)),9)
        np.testing.assert_array_equal(zero.initialize(h),w.initialize(h))

    def test_shrinkage_weighting_and_empty(self):
        r=np.array([[1.,0.],[-1.,0.],[0.,3.],[0.,-3.]])
        g=np.array([0,0,1,1]);weights=np.ones(4);base=np.eye(2)
        c=grouped_covariances(r,weights,g,base)
        np.testing.assert_allclose(c[0],np.diag([1.,.1]))
        np.testing.assert_allclose(c[1],np.diag([.1,8.2]))
        np.testing.assert_array_equal(c[2],base)
        duplicate=grouped_covariances(np.repeat(r,2,axis=0),np.repeat(weights/2,2),np.repeat(g,2),base)
        np.testing.assert_allclose(c,duplicate)


if __name__=='__main__':unittest.main()
