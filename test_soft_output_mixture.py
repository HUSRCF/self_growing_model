import unittest
import numpy as np
from spectral_kernel_energy import statistics, spectral_cost, kernel_moments, blocked_value
from soft_output_mixture import terms_from_statistics, value_gradient, interval_grid_difference, exact_point_terms


class SoftOutputMixtureTests(unittest.TestCase):
    def test_exact_point_variant_against_direct_noise_quadrature(self):
        size = 257; angle = 2*np.pi*np.arange(size)/size
        noise = np.stack(np.meshgrid(angle, angle, indexing='ij'), -1)
        mass = np.exp(2*np.cos(noise[..., 0])+4*np.cos(noise[..., 1])); mass /= mass.sum()
        difference_mass = np.fft.ifft2(np.abs(np.fft.fft2(mass))**2).real
        x = np.array([[.13,.41],[1.27,-.88],[-2.19,2.06]]); y = np.array([.51,-1.02])
        def distance(delta):
            return np.sqrt(np.maximum((2-2*np.cos(delta)).sum(-1),0))
        a0 = distance(x-y).mean()
        b0 = sum(distance(x[i]-x[j]) for i in range(3) for j in range(3) if i!=j)/6
        a1 = np.mean([np.sum(mass*distance(noise+point-y)) for point in x])
        b1 = sum(np.sum(mass*distance(noise+x[i]-x[j])) for i in range(3) for j in range(3) if i!=j)/6
        b2 = sum(np.sum(difference_mass*distance(noise+x[i]-x[j])) for i in range(3) for j in range(3) if i!=j)/6
        row = blocked_value(x,y,np.zeros(3,bool),size,np.log([2.,4.]),mixture=True)
        for alpha in [0.,.37,1.]:
            direct = (1-alpha)*a0+alpha*a1-.5*((1-alpha)**2*b0+2*alpha*(1-alpha)*b1+alpha**2*b2)
            np.testing.assert_allclose(value_gradient(exact_point_terms(row),alpha)[0],direct,atol=1e-5,rtol=0)

    def test_direct_independent_discrete_noise(self):
        size = 9; angle = 2*np.pi*np.arange(size)/size
        indices = np.array([[1, 3], [5, 2], [7, 8]])
        centers = angle[indices]; truth = angle[np.array([2, 4])]
        failed = np.array([False, True, False])
        s = statistics(centers[None], truth[None], failed[None], size)
        mass = np.exp(1.3*np.cos(angle[:, None])+2.1*np.cos(angle[None])); mass /= mass.sum()
        kernel = np.fft.fft2(mass).real; terms = terms_from_statistics(s, kernel)
        grid = np.stack(np.meshgrid(angle, angle, indexing='ij'), -1).reshape(-1, 2)
        distance = np.sqrt(np.maximum((2-2*np.cos(grid[:, None]-grid[None])).sum(-1), 0))
        target_distance = np.sqrt(np.maximum((2-2*np.cos(grid-truth)).sum(-1), 0))
        for alpha in [0., .17, .5, 1.]:
            noise = alpha*mass.copy(); noise[0, 0] += 1-alpha
            distributions = np.array([np.roll(noise, tuple(index), axis=(0, 1)).ravel() for index in indices])
            attraction = np.mean(distributions@target_distance)
            pair = sum(distributions[i]@distance@distributions[j] for i in range(3) for j in range(3) if i!=j)/6
            np.testing.assert_allclose(value_gradient(terms, alpha)[0], attraction-.5*pair+2/3, atol=1e-13)
        # Choosing one mode for the entire ensemble loses mixed-particle cross terms.
        endpoint = value_gradient(terms, 1)[0]
        true = value_gradient(terms, .5)[0]
        naive = .5*terms['baseline']+.5*endpoint
        np.testing.assert_allclose(true-naive, -.25*terms['quadratic'], atol=1e-14)
        self.assertGreater(float(np.abs(true-naive)[0]), 1e-5)

    def test_blocked_terms_endpoints_and_derivative(self):
        rng = np.random.default_rng(24)
        x, y = rng.normal(size=(5, 2)), rng.normal(size=2); f = np.zeros(5, bool)
        theta = np.log([30., 2.])
        s = statistics(x[None], y[None], f[None], 33)
        terms = terms_from_statistics(s, kernel_moments(s['modes'], theta)[0])
        row = blocked_value(x, y, f, 33, theta, mixture=True)
        for key in ['linear', 'quadratic', 'baseline']:
            np.testing.assert_allclose(row[key], terms[key][0], atol=1e-14)
        self.assertEqual(float(value_gradient(row, 0)[0]), row['baseline'])
        np.testing.assert_allclose(value_gradient(row, 1)[0], row['cost'], atol=1e-14)
        corrected = exact_point_terms(row)
        self.assertEqual(float(value_gradient(corrected, 0)[0]), row['baseline'])
        np.testing.assert_allclose(value_gradient(corrected, 1)[0], row['raw_cost'], atol=1e-14)
        np.testing.assert_allclose(value_gradient(corrected, 1)[0]-value_gradient(row, 1)[0], row['baseline_error'], atol=1e-14)
        alpha, eps = .37, 1e-6
        fd = (value_gradient(row, alpha+eps)[0]-value_gradient(row, alpha-eps)[0])/(2*eps)
        np.testing.assert_allclose(fd, value_gradient(row, alpha)[1], atol=1e-9)
        fd = (value_gradient(corrected, alpha+eps)[0]-value_gradient(corrected, alpha-eps)[0])/(2*eps)
        np.testing.assert_allclose(fd, value_gradient(corrected, alpha)[1], atol=1e-9)
        for bad in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                value_gradient(row, bad)

    def test_uniform_interval_bound(self):
        first = dict(baseline=np.zeros(4), linear=np.array([1., 0., 3., -2.]), quadratic=np.array([-2., 1., 0., 2.]))
        second = dict(baseline=np.zeros(4), linear=np.zeros(4), quadratic=np.zeros(4))
        bounds = interval_grid_difference(first, second)
        values, gradients = value_gradient(first, np.linspace(0, 1, 10001)[:, None])
        np.testing.assert_allclose(bounds['cost'], np.abs(values).max(0), atol=1e-8)
        np.testing.assert_allclose(bounds['gradient'], np.abs(gradients).max(0), atol=1e-14)
