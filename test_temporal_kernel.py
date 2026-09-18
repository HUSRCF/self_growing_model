import unittest
import numpy as np
from spectral_kernel_energy import statistics,spectral_cost,kernel_moments
from fit_temporal_kernel import raw_objective


class TemporalKernelTests(unittest.TestCase):
    def test_raw_cost_and_gradient(self):
        rng=np.random.default_rng(37)
        s=statistics(rng.normal(size=(1,5,2)),rng.normal(size=(1,2)),np.zeros((1,5),bool),33)
        theta=np.log([3.,7.]);value,gradient=raw_objective(theta,s)
        np.testing.assert_allclose(value,spectral_cost(s,kernel_moments(s['modes'],theta)[0])[0],atol=1e-13,rtol=0)
        eps=1e-5
        finite=[(raw_objective(theta+eps*v,s)[0]-raw_objective(theta-eps*v,s)[0])/(2*eps) for v in np.eye(2)]
        np.testing.assert_allclose(gradient,finite,atol=1e-8,rtol=1e-6)
