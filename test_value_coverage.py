import unittest
import numpy as np
from audit_crossfit_value import crossfit
from audit_value_coverage import predict_crossfit, extra_windows


class CoverageTests(unittest.TestCase):
    def test_original_reduction_and_label_isolation(self):
        rng = np.random.default_rng(87)
        x = rng.normal(size=(12, 8, 5)); y = rng.normal(size=(12, 8)); v = np.repeat([1, 2, 3], 4)
        a = predict_crossfit(x, y, v, x, v)
        np.testing.assert_allclose(a, crossfit(x, y, v), atol=1e-12)
        for contextual in (True, False):
            a = predict_crossfit(x, y, v, x, v, contextual)
            yy = y.copy(); yy[v==1] += rng.normal(size=(4, 8))*100
            b = predict_crossfit(x, yy, v, x, v, contextual)
            np.testing.assert_array_equal(a[v==1], b[v==1])

    def test_action_only_and_extra_starts(self):
        x = np.zeros((6, 8, 2)); v = np.repeat([1, 2, 3], 2)
        y = np.tile(np.arange(8), (6, 1))
        scores = predict_crossfit(x, y, v, x, v, False)
        np.testing.assert_array_equal(scores, np.tile(np.arange(8)-3.5, (6, 1)))
        class Pool:
            videos = [1]
            pool = {1: {'starts': np.arange(63, 100), 'y': np.zeros((500, 2))}}
        old = {'video': np.array([1, 1]), 'start': np.array([63, 64])}
        w = extra_windows(Pool(), old, count=24)
        self.assertEqual(len(np.unique(w['start'])), 24)
        self.assertFalse(np.isin(w['start'], old['start']).any())
        self.assertEqual(w['truth'].shape, (24, 300, 2))
