import unittest
import numpy as np
from audit_matched_value_targets import fit_policy


class MatchedValueTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(9, 8, 5)); y = rng.normal(size=(9, 8)); v = np.repeat([1, 2, 3], 3)
        pe = rng.dirichlet(np.ones(2), size=9)
        tr = rng.dirichlet(np.ones(8), size=(9, 2))
        return x, y, v, pe, tr

    def test_held_labels_cannot_change_scores_or_calibration(self):
        x, y, v, pe, tr = self.data(); a = fit_policy(x, y, v, pe, tr)
        changed = y.copy(); changed[v==1] *= -100
        b = fit_policy(x, changed, v, pe, tr)
        for key in ['scores', 'probability', 'kl']:
            np.testing.assert_array_equal(a[key][v==1], b[key][v==1])
        self.assertEqual(a['folds']['1'], b['folds']['1'])

    def test_positive_target_scale_keeps_policy(self):
        x, y, v, pe, tr = self.data()
        a, b = fit_policy(x, y, v, pe, tr), fit_policy(x, 4*y, v, pe, tr)
        np.testing.assert_allclose(b['scores'], 4*a['scores'], atol=1e-12)
        np.testing.assert_allclose(a['probability'], b['probability'], atol=1e-12)
        for video in a['folds']:
            self.assertFalse(a['folds'][video]['capped'])
            self.assertFalse(b['folds'][video]['capped'])
