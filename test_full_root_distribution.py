import itertools
import unittest
import numpy as np
from audit_full_root_distribution import mixture_terms, mixture_score, change_terms
from ensemble_score_objective import energy_costs


class FullRootTests(unittest.TestCase):
    def test_exact_independent_root_enumeration(self):
        rng = np.random.default_rng(281)
        x = rng.normal(size=(1, 2, 3, 4)); y = rng.normal(size=(1, 4))
        f = rng.random((1, 2, 3)) < .3; w = np.array([[.3, .7]])
        a, b = mixture_terms(x, y, f); exact = 0.
        for roots in itertools.product(range(2), repeat=3):
            probability = np.prod(w[0, list(roots)])
            selected = x[:, list(roots), np.arange(3)]
            failures = f[:, list(roots), np.arange(3)]
            exact += probability*energy_costs(selected, y, failures)[0]
        np.testing.assert_allclose(mixture_score(a, b, w), exact, atol=1e-14)
        new = np.array([[.8, .2]])
        linear, quadratic = change_terms(a, b, new, w)
        np.testing.assert_allclose(mixture_score(a, b, new)-mixture_score(a, b, w), linear+quadratic, atol=1e-14)
        mixed = .75*w+.25*new
        mixed_linear, mixed_quadratic = change_terms(a, b, mixed, w)
        np.testing.assert_allclose(mixed_linear, .25*linear, atol=1e-14)
        np.testing.assert_allclose(mixed_quadratic, .25**2*quadratic, atol=1e-14)

    def test_identical_components_and_shared_index_exclusion(self):
        points = np.array([0., 1., 3.]).reshape(1, 1, 3, 1)
        x = np.repeat(points, 2, axis=1); y = np.zeros((1, 1)); f = np.zeros((1, 2, 3), bool)
        a, b = mixture_terms(x, y, f)
        self.assertAlmostEqual(b[0, 0, 1], 2.)
        for w in (np.array([[1., 0.]]), np.array([[.4, .6]])):
            np.testing.assert_allclose(mixture_score(a, b, w), energy_costs(x[:, 0], y, f[:, 0])[0])
        with self.assertRaises(ValueError): mixture_score(a, b, np.array([[1., 1.]]))
