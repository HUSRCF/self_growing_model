import unittest
from unittest.mock import patch
import numpy as np
from confirm_temporal_kernel_holdout import hold_windows
from v20_rnn_mixture.engine.common import SPLITS


class TemporalKernelHoldoutTests(unittest.TestCase):
    def test_tail_reads_only_hold_videos_and_preserves_layout(self):
        y=np.arange(8000).reshape(4000,2)
        with patch('confirm_temporal_kernel_holdout.load_video',return_value=y) as read:
            w=hold_windows('tail')
        self.assertEqual([c.args[0] for c in read.call_args_list],SPLITS['train'][-3:])
        self.assertEqual(w['history'].shape,(24,32,2))
        self.assertEqual(w['truth'].shape,(24,300,2))
        np.testing.assert_array_equal(w['history'][0],y[1968:2000])
        np.testing.assert_array_equal(w['truth'][0],y[2000:2300])
        self.assertEqual(w['start'][-1],3699)
        np.testing.assert_array_equal(w['truth'][-1],y[3700:4000])

    def test_invalid_region(self):
        with self.assertRaises(ValueError):hold_windows('unknown')
