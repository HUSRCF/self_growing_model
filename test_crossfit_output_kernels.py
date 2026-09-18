import unittest
import numpy as np
from crossfit_output_kernels import fold_partition, combine_statistics
from spectral_kernel_energy import statistics, value_gradient


class CrossfitKernelTests(unittest.TestCase):
    def test_threshold_excludes_held_video(self):
        video = np.array([1, 1, 2, 2, 3, 3])
        motion = np.arange(6, dtype=float)
        train, threshold, bins = fold_partition(motion, video, 3)
        motion[~train] = -1000
        other_train, other_threshold, other_bins = fold_partition(motion, video, 3)
        self.assertEqual(threshold, other_threshold)
        np.testing.assert_array_equal(bins[train], other_bins[other_train])
        self.assertEqual(int(train.sum()), 4)

    def test_group_pooling_matches_full_statistics(self):
        rng = np.random.default_rng(82)
        x, y = rng.normal(size=(7, 4, 2)), rng.normal(size=(7, 2))
        f = np.zeros((7, 4), dtype=bool)
        groups = []
        for sl in [slice(0, 2), slice(2, 7)]:
            s = statistics(x[sl], y[sl], f[sl], 17)
            for key in ['attraction', 'pair', 'baseline', 'penalty']:
                s[key] = s[key].mean(axis=0, keepdims=True)
            groups.append(s)
        pooled = combine_statistics(groups, [2, 5])
        batch = statistics(x, y, f, 17)
        for theta in [None, np.log([3., 23.])]:
            v, g = value_gradient(batch, theta); pv, pg = value_gradient(pooled, theta)
            np.testing.assert_allclose(pv[0], v.mean(), atol=1e-14)
            np.testing.assert_allclose(pg[0], g.mean(0), atol=1e-14)
