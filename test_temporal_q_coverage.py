import unittest
from unittest.mock import patch
import numpy as np
from train_temporal_q_coverage import prefix_half,assert_train_boundary


class TemporalQTests(unittest.TestCase):
    def test_selection_is_fixed_and_balanced(self):
        v=np.repeat([9,1],8);ids=prefix_half(v)
        np.testing.assert_array_equal(ids,[0,1,2,3,8,9,10,11])
        with self.assertRaises(ValueError):prefix_half(v[:-1])

    def test_training_target_boundary(self):
        with patch('train_temporal_q_coverage.load_video',return_value=np.zeros((1000,2))):
            assert_train_boundary(dict(video=np.array([1]),start=np.array([449])))
            with self.assertRaises(ValueError):assert_train_boundary(dict(video=np.array([1]),start=np.array([450])))


if __name__=='__main__':unittest.main()
