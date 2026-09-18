import unittest
import numpy as np
from fit_energy_output_kernel import mean_statistics
from spectral_kernel_energy import statistics, value_gradient


class AggregateTests(unittest.TestCase):
    def test_linear_aggregation(self):
        rng = np.random.default_rng(52)
        x, y = rng.normal(size=(5, 4, 2)), rng.normal(size=(5, 2))
        failed = rng.random((5, 4)) < .2
        batch = statistics(x, y, failed, 33)
        aggregate = mean_statistics(x, y, failed, 33)
        for theta in [None, np.log([10., 30.]), np.log([.001, 1e6])]:
            value, gradient = value_gradient(batch, theta)
            av, ag = value_gradient(aggregate, theta)
            np.testing.assert_allclose(av[0], value.mean(), atol=1e-14)
            np.testing.assert_allclose(ag[0], gradient.mean(0), atol=1e-14)
