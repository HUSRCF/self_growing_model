import unittest
import numpy as np
from audit_initial_velocity_rng import crossed_summary


class CrossedTests(unittest.TestCase):
    def test_additive_and_total(self):
        x=np.arange(4.)[:,None]+np.arange(3.)[None,:]*2
        s=crossed_summary(x)
        self.assertAlmostEqual(s['sum_squares']['interaction'],0.)
        self.assertAlmostEqual(sum(s['sum_squares'].values()),s['total_sum_squares'])
        x=x.copy();x[0,0]+=3
        s=crossed_summary(x)
        self.assertAlmostEqual(sum(s['sum_squares'].values()),s['total_sum_squares'])
        self.assertGreater(s['sum_squares']['interaction'],0)
        t=crossed_summary(x.T)
        self.assertAlmostEqual(t['sum_squares']['action'],s['sum_squares']['initialization'])
        self.assertEqual(crossed_summary(np.ones((2,2)))['total_sum_squares'],0.)
        with self.assertRaises(ValueError):crossed_summary(np.ones((1,2)))


if __name__=='__main__':unittest.main()
