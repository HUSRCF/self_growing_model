import unittest
import numpy as np
from spectral_kernel_energy import statistics, spectral_cost, value_gradient, kernel_moments


class SpectralKernelTests(unittest.TestCase):
    def test_discrete_noise_direct_integration(self):
        size = 9; angle = 2*np.pi*np.arange(size)/size
        indices = np.array([[1, 3], [5, 2], [7, 8]])
        centers = angle[indices][None]; truth = angle[np.array([[2, 4]])]
        failed = np.array([[False, True, False]])
        stats = statistics(centers, truth, failed, size)
        mass = np.exp(1.3*np.cos(angle[:, None])+2.1*np.cos(angle[None]))
        mass /= mass.sum()
        kernel = np.fft.fft2(mass).real
        grid = np.stack(np.meshgrid(angle, angle, indexing='ij'), -1).reshape(-1, 2)
        distribution = np.array([np.roll(mass, tuple(index), axis=(0, 1)).ravel() for index in indices])
        distance = np.sqrt(np.maximum((2-2*np.cos(grid[:, None]-grid[None])).sum(-1), 0))
        attraction = (distribution*np.sqrt(np.maximum((2-2*np.cos(grid-truth[0])).sum(-1), 0))).sum()/3
        pair = sum(distribution[i]@distance@distribution[j] for i in range(3) for j in range(3) if i != j)/6
        np.testing.assert_allclose(spectral_cost(stats, kernel), attraction-.5*pair+2/3, atol=1e-13)

    def test_analytic_gradient_zero_and_periodicity(self):
        rng = np.random.default_rng(33); c = rng.normal(size=(4, 5, 2)); y = rng.normal(size=(4, 2)); f = np.zeros((4, 5), bool)
        stats = statistics(c, y, f, 33); theta = np.log([3., 17.])
        value, gradient = value_gradient(stats, theta)
        np.testing.assert_array_equal(value_gradient(stats)[0], stats['baseline'])
        for i in range(2):
            d = np.eye(2)[i]*1e-5
            fd = (value_gradient(stats, theta+d)[0]-value_gradient(stats, theta-d)[0])/(2e-5)
            np.testing.assert_allclose(gradient[:, i], fd, atol=1e-8, rtol=1e-6)
        other = statistics(c+2*np.pi, y-4*np.pi, f, 33)
        np.testing.assert_allclose(value, value_gradient(other, theta)[0], atol=1e-13)

    def test_bessel_moments_against_circular_quadrature(self):
        grid = 2*np.pi*np.arange(4096)/4096; modes = np.arange(-4, 5)
        kernel, _ = kernel_moments(modes, np.log([.7, 350.]))
        values = []
        for k in [.7, 350.]:
            mass = np.exp(k*(np.cos(grid)-1)); mass /= mass.sum()
            values.append(np.cos(modes[:, None]*grid)@mass)
        np.testing.assert_allclose(kernel, np.outer(*values), atol=1e-13)
