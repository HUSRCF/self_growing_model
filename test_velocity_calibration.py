import unittest
import numpy as np
from audit_velocity_calibration import residual_stats, causal_groups


class CalibrationTests(unittest.TestCase):
    def test_gaussian_coverage_and_weights(self):
        r=np.random.default_rng(9).normal(size=(100000,2));cov=np.eye(2)
        s=residual_stats(r,np.ones(len(r)),cov)
        self.assertAlmostEqual(s['mean_mahalanobis_squared'],2,delta=.02)
        self.assertAlmostEqual(s['coverage']['0.9'],.9,delta=.005)
        small=r[:20];w=np.arange(1,21.)
        a=residual_stats(small,w,cov);b=residual_stats(np.repeat(small,2,axis=0),np.repeat(w/2,2),cov)
        np.testing.assert_allclose(a['covariance'],b['covariance'])
        self.assertAlmostEqual(a['rmse'],b['rmse'])
        self.assertIsNone(residual_stats(np.empty((0,2)),np.array([]),cov))

    def test_groups_do_not_read_future(self):
        class Base:
            def state_from_history(self,h):return (h[:,-1,0]>0).astype(int),None
        y=np.random.default_rng(3).normal(size=(100,2));changed=y.copy();changed[61:]+=99
        q,s=causal_groups(Base(),y);q2,s2=causal_groups(Base(),changed)
        # t=32..60 inclusive precede the changed future.
        np.testing.assert_array_equal(q[:29],q2[:29]);np.testing.assert_array_equal(s[:29],s2[:29])


if __name__=='__main__':unittest.main()
