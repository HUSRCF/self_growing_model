import unittest
import numpy as np
from audit_crossfit_value import crossfit, trajectory_cost


class CrossfitValueTests(unittest.TestCase):
    def test_held_video_labels_do_not_affect_predictions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 8, 5)); y = rng.normal(size=(12, 8))
        v = np.repeat(np.arange(3), 4)
        a = crossfit(x, y, v)
        changed = y.copy(); changed[v==0] += rng.normal(size=(4, 8))*100
        b = crossfit(x, changed, v)
        np.testing.assert_array_equal(a[v==0], b[v==0])
        np.testing.assert_allclose(a.mean(1), 0, atol=1e-14)

    def test_cost_units_and_failure(self):
        truth = np.zeros((2, 300, 2)); p = np.zeros((2, 4, 300, 2))
        f = np.zeros((2, 4, 300), bool)
        np.testing.assert_array_equal(trajectory_cost(p, truth, f), 0)
        p[:] = np.pi; f[1] = True
        np.testing.assert_allclose(trajectory_cost(p, truth, f), [2, 4])

    def test_window_offsets_and_candidate_permutation(self):
        rng = np.random.default_rng(81)
        x = rng.normal(size=(12, 8, 5)); y = rng.normal(size=(12, 8))
        v = np.repeat(np.arange(3), 4)
        a = crossfit(x, y, v)
        b = crossfit(x, y+rng.normal(size=(12, 1)), v)
        np.testing.assert_allclose(a, b, atol=1e-12)
        order = rng.permutation(8)
        c = crossfit(x[:, order], y[:, order], v)
        np.testing.assert_allclose(c, a[:, order], atol=1e-12)
