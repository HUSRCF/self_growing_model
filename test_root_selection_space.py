import unittest
import numpy as np
from audit_root_selection_space import candidate_costs, cross_stream
from audit_full_root_distribution import mixture_score


class RootSelectionSpaceTests(unittest.TestCase):
    def test_candidates(self):
        p = np.array([[.2, .8], [.5, .5]])
        a = np.array([[1., 2.], [3., 1.]])
        b = np.array([[[0., 1.], [1., 0.]]]*2)
        cost, policies = candidate_costs(a, b, p)
        np.testing.assert_array_equal(policies[:, 0], p)
        np.testing.assert_allclose(policies.sum(-1), 1.)
        for i in range(3):
            np.testing.assert_array_equal(cost[:, i], mixture_score(a, b, policies[:, i]))

    def test_held_stream_not_used_for_selection(self):
        c = np.array([[[0., -5., 1.]], [[0., 2., -1.]], [[0., 2., -1.]]])
        rows = cross_stream(c)
        self.assertEqual(rows[0]['selected'], [2])
        self.assertEqual(rows[0]['delta'], [1.])
        self.assertEqual(rows[0]['same_stream_delta'], [-5.])
        c[0] = [0., 100., -100.]
        self.assertEqual(cross_stream(c)[0]['selected'], [2])
        with self.assertRaises(ValueError):
            cross_stream(c[:1])
