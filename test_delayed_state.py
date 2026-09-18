import unittest
from types import SimpleNamespace
import numpy as np
from test_delayed_root import ToyMachine
from audit_crossfit_value import continuation
from audit_delayed_state import correlation


class DelayedStateTests(unittest.TestCase):
    def test_pre_read_snapshot_and_no_alias(self):
        engine=SimpleNamespace(machine=ToyMachine(),base=SimpleNamespace(execute_rule=lambda h,q,r:h[:,-1]+.001))
        h=np.zeros((1,32,2));snap={0:None,50:None}
        p,f=continuation(engine,h,10,particles=3,snapshots=snap)
        plain,pf=continuation(engine,h,10,particles=3)
        np.testing.assert_array_equal(p,plain);np.testing.assert_array_equal(f,pf)
        np.testing.assert_array_equal(snap[50]['hidden'],50.)
        np.testing.assert_array_equal(snap[50]['history'],p[:,:,18:50].reshape(3,32,2))
        snap[50]['history'][:]=99
        np.testing.assert_array_equal(p,plain);np.testing.assert_array_equal(h,0.)
        with self.assertRaises(ValueError):continuation(engine,h,10,snapshots={300:None})

    def test_within_video_correlation(self):
        self.assertIsNone(correlation([1,1],[2,3]))
        self.assertAlmostEqual(correlation([0,1,10,11],[0,2,-10,-8],np.array([1,1,2,2])),1.)
