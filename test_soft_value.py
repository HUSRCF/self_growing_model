import unittest
import numpy as np
from audit_soft_value import tilt, calibrate, conditional_kl


class SoftValueTests(unittest.TestCase):
    def test_identity_support_event_mass_and_shift(self):
        t = np.array([[[.2, .8, 0], [.1, .3, .6]]]); pe = np.array([[.3, .7]])
        s = np.array([[1., 2., 3.]])
        np.testing.assert_array_equal(tilt(t, s, 0), t)
        q = tilt(t, s, 2)
        np.testing.assert_allclose((pe[:, :, None]*q).sum(-1), pe)
        self.assertEqual(q[0, 0, 2], 0)
        np.testing.assert_allclose(q, tilt(t, s+100, 2))
        self.assertLess(float((pe[:, :, None]*q*s[:, None]).sum()), float((pe[:, :, None]*t*s[:, None]).sum()))

    def test_calibration_and_constant_scores(self):
        t = np.array([[[.2, .8], [.7, .3]], [[.5, .5], [.4, .6]]])
        pe = np.array([[.3, .7], [.6, .4]]); s = np.array([[1., 2.], [2., 1.]])
        strength, kl, capped = calibrate(pe, t, s)
        self.assertFalse(capped); self.assertAlmostEqual(kl, .01, places=12)
        self.assertAlmostEqual(conditional_kl(pe, t, tilt(t, s, strength)).mean(), .01, places=12)
        _, kl, capped = calibrate(pe, t, np.ones_like(s))
        self.assertTrue(capped); self.assertAlmostEqual(kl, 0, places=12)
        with self.assertRaises(ValueError): tilt(t, s, -1)
