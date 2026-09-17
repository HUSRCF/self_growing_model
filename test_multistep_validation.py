import unittest
import numpy as np
from validate_multistep_initializer import paired_summary


class ValidationTests(unittest.TestCase):
    def test_paired_statistics(self):
        s=paired_summary([-1,1,-1,1]);self.assertEqual(s['mean'],0.)
        self.assertAlmostEqual(s['conditional_rng_se'],1/np.sqrt(3))
        self.assertEqual((s['better'],s['worse']),(2,2))
        self.assertEqual(paired_summary([0,0])['conditional_rng_se'],0.)
        with self.assertRaises(ValueError):paired_summary([1])


if __name__=='__main__':unittest.main()
