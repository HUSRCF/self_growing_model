import unittest
import numpy as np
from audit_writer_temporal_shift import motion,select_videos
from v20_rnn_mixture.engine.common import DT


class TemporalShiftTests(unittest.TestCase):
    def test_motion_uses_history_increments(self):
        h=np.zeros((2,32,2));h[1]=np.arange(32)[:,None]*DT
        np.testing.assert_allclose(motion(h),[0,1],atol=1e-14)

    def test_selection_preserves_row_alignment(self):
        w=dict(video=np.array([1,2,3,2]),start=np.arange(4),history=np.arange(8).reshape(4,2))
        out=select_videos(w,[2]);np.testing.assert_array_equal(out['start'],[1,3]);np.testing.assert_array_equal(out['history'],[[2,3],[6,7]])


if __name__=='__main__':unittest.main()
