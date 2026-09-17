import unittest
import numpy as np
from audit_source_q_gradient import stability,cosine


class GradientStabilityTests(unittest.TestCase):
    def test_scale_and_zero(self):
        x=np.array([[1,2],[2,1],[-1,1],[3,2]],float)
        a=stability(x);b=stability(x*5)
        self.assertAlmostEqual(a['mean_to_rms_se'],b['mean_to_rms_se'])
        self.assertAlmostEqual(a['pairwise_cosine_mean'],b['pairwise_cosine_mean'])
        self.assertAlmostEqual(b['variance_trace'],25*a['variance_trace'])
        self.assertIsNone(cosine([0,0],[1,2]));self.assertIsNone(stability(np.zeros((4,2)))['mean_to_rms_se'])

    def test_opposite_halves(self):
        r=stability([[1,0],[1,0],[-1,0],[-1,0]])
        self.assertEqual(r['first_half_vs_second_half_cosine'],-1)
        self.assertEqual(r['negative_pair_count'],4)
        self.assertEqual(r['mean_to_rms_se'],0)


if __name__=='__main__':unittest.main()
