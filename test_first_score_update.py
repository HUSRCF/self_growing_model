import numpy as np
import unittest
from audit_first_score_update import first_step, candidates


def check_first_adam_step():
    g = np.linspace(-.2, .2, 16).reshape(8, 2)
    np.testing.assert_allclose(first_step(g), -.1*g/(np.abs(g)+1e-8), rtol=1e-14)
    np.testing.assert_array_equal(first_step(np.zeros((8, 2))), 0.)
    np.testing.assert_array_equal(first_step(-g), -first_step(g))
    with np.testing.assert_raises(ValueError):
        first_step(np.full((8, 2), np.nan))


def check_budget_and_control_split():
    a, b, s = [np.full((10, 8, 2), v).tolist() for v in [1., 3., 2.]]
    report = {'phases': {'fit': [{'gradients': a} for _ in range(8)],
                         'evaluation': [{'gradients': b, 'score_gradients': s} for _ in range(8)]},
              'coefficients': [.5]*10}
    theta, gradients = candidates(report)
    for name, value in [('raw8', 1.), ('raw16', 2.), ('controlled16', 2.), ('reverse_raw16', -2.)]:
        np.testing.assert_array_equal(gradients[name], value)
    np.testing.assert_array_equal(theta['zero'], 0.)


class FirstUpdateTests(unittest.TestCase):
    def test_first_adam_step(self):
        check_first_adam_step()

    def test_budget_and_control_split(self):
        check_budget_and_control_split()
