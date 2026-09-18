import unittest
import numpy as np
from periodic_output_kernel import nll_gradient, fit_kernel, convolve


class PeriodicKernelTests(unittest.TestCase):
    def test_duplicate_centers_preserve_density_normalization(self):
        rng = np.random.default_rng(821)
        c = rng.normal(size=(7, 5, 2)); y = rng.normal(size=(7, 2)); theta = np.log([3., 10.])
        loss, gradient = nll_gradient(theta, c, y)
        copied_loss, copied_gradient = nll_gradient(theta, np.concatenate([c]*4, axis=1), y)
        self.assertAlmostEqual(loss, copied_loss, places=12)
        np.testing.assert_allclose(gradient, copied_gradient, atol=1e-12)

    def test_gradient_and_periodicity(self):
        rng = np.random.default_rng(92)
        c = rng.normal(size=(7, 5, 2)); y = rng.normal(size=(7, 2)); theta = np.log([3., 10.])
        loss, gradient = nll_gradient(theta, c, y)
        for i in range(2):
            d = np.eye(2)[i]*1e-5
            fd = (nll_gradient(theta+d, c, y)[0]-nll_gradient(theta-d, c, y)[0])/(2e-5)
            self.assertAlmostEqual(fd, gradient[i], places=7)
        self.assertAlmostEqual(loss, nll_gradient(theta, c+2*np.pi, y-4*np.pi)[0], places=12)

    def test_density_normalization(self):
        # Single kernel, product density: integrate one angle on a periodic grid.
        from scipy.special import i0e
        grid = np.linspace(-np.pi, np.pi, 4096, endpoint=False); k = 12.
        density = np.exp(k*(np.cos(grid)-1))/(2*np.pi*i0e(k))
        self.assertAlmostEqual(float(density.mean()*2*np.pi), 1., places=12)

    def test_zero_and_copy_and_fit(self):
        p = np.zeros((3, 8, 3, 2)); original = p.copy()
        np.testing.assert_array_equal(convolve(p, None, 1), p)
        changed = convolve(p, np.ones((3, 2))*10, 1)
        np.testing.assert_array_equal(p, original); self.assertTrue(np.any(changed != p))
        rng = np.random.default_rng(81)
        c = np.zeros((100, 4, 2)); y = rng.vonmises(0, [4., 8.], size=(100, 2))
        fit = fit_kernel(c, y)
        self.assertTrue(np.isfinite(fit['nll'])); self.assertFalse(any(fit['at_boundary']))
