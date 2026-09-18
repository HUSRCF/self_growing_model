import unittest
import numpy as np
from confirm_root_mixture_holdout import summarize_runs


class ConfirmationTests(unittest.TestCase):
    def test_paired_summary_uses_actual_seed_count(self):
        rows = [dict(cost=1.+d, delta=d, per_video_delta={'13': d}, horizon_costs=[1.+d]*3)
                for d in [-.2, -.1, 0., .1, .2, .3, .4, .5]]
        r = summarize_runs(rows); delta = np.array([v['delta'] for v in rows])
        self.assertEqual(r['better_seeds'], 2)
        self.assertAlmostEqual(r['conditional_seed_se'], delta.std(ddof=1)/np.sqrt(8))
        self.assertAlmostEqual(r['delta'], delta.mean())
        with self.assertRaises(ValueError): summarize_runs(rows[:1])
