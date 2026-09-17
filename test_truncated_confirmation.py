import unittest
import numpy as np
from confirm_truncated_writer import paired,window_costs


class ConfirmationTests(unittest.TestCase):
    def test_paired_not_unpaired(self):
        r=paired([11,21,31],[10,20,30]);self.assertEqual(r['delta'],1);self.assertEqual(r['conditional_seed_se'],0)
        with self.assertRaises(ValueError):paired([1],[1])

    def test_identical_truth_particles_only_failure_cost(self):
        truth=np.zeros((2,300,2));pred=np.zeros((2,4,300,2));failed=np.zeros((2,4,300),bool)
        failed[1,0]=True
        np.testing.assert_allclose(window_costs(pred,truth,failed),[0,.5],atol=1e-14)


if __name__=='__main__':unittest.main()
