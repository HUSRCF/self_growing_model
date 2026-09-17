import unittest
import numpy as np
from audit_writer_ranking import ranking_summary


class WriterRankingTests(unittest.TestCase):
    def test_winner_reversal_and_independent_groups(self):
        fresh=np.array([[1,.8,1.2]]*3+[[1,1.3,.9]]*3)
        result=ranking_summary([1,.5,1.1],fresh)
        self.assertEqual(result['original_chosen'],1)
        self.assertEqual(result['same_winner_fraction'],.5)
        self.assertAlmostEqual(result['chosen_mean_difference'],.05)
        self.assertEqual(result['group3_winners'],[1,2])
        np.testing.assert_allclose(result['group3_cross_evaluation'],[.3,.2])

    def test_incumbent_selection_has_zero_paired_change(self):
        z=ranking_summary([0,1,2],np.arange(18).reshape(6,3))
        self.assertEqual(z['chosen_mean_difference'],0)
        self.assertEqual(z['chosen_conditional_rng_se'],0)


if __name__=='__main__':unittest.main()
