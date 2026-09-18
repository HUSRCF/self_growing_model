import unittest
import numpy as np
from audit_gate_temporal_coverage import region_starts, nearest_other_video


class TemporalCoverageTests(unittest.TestCase):
    def test_regions_and_exclusion(self):
        excluded = [63, 100, 1200]
        p = region_starts(4000, 'prefix', excluded)
        t = region_starts(4000, 'tail', excluded)
        self.assertEqual(len(set(p)), 64)
        self.assertEqual(len(set(t)), 64)
        self.assertFalse(set(p) & set(excluded))
        self.assertTrue(np.all(p+300 < 2000))
        self.assertTrue(np.all(t-31 >= 2000))
        self.assertTrue(np.all(t+300 < 4000))

    def test_reference_excludes_same_video(self):
        actual = nearest_other_video(np.array([[0., 0.]]), np.array([[0., 0.], [3., 4.]]), [1], [1, 2])
        np.testing.assert_allclose(actual, [np.sqrt(12.5)])
        with self.assertRaises(ValueError):
            nearest_other_video(np.zeros((1, 2)), np.zeros((1, 2)), [1], [1])

    def test_invalid_region_or_short_video(self):
        with self.assertRaises(ValueError):
            region_starts(4000, 'unknown')
        with self.assertRaises(ValueError):
            region_starts(100, 'tail')
