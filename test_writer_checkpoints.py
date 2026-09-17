import unittest
import numpy as np
from audit_writer_checkpoints import summarize_checkpoints


class CheckpointAuditTests(unittest.TestCase):
    def test_cross_evaluation_does_not_use_own_selected_half(self):
        c=np.array([[1,.8,1.1,1.2]]*4+[[1,1.3,.9,1.2]]*4)
        r=summarize_checkpoints(c,1)
        self.assertEqual(r['half_selected_indices'],[1,2])
        np.testing.assert_allclose(r['half_cross_deltas'],[.3,.1])
        self.assertAlmostEqual(r['original_selected_delta'],.05)
        self.assertEqual(r['original_selected_better_seeds'],4)

    def test_zero_selected_delta_is_exactly_zero(self):
        r=summarize_checkpoints(np.arange(32).reshape(8,4),0)
        self.assertEqual(r['original_selected_delta'],0)
        self.assertEqual(r['conditional_rng_se'][0],0)


if __name__=='__main__':unittest.main()
