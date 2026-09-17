import unittest
import numpy as np
from score_control_variate import fit_coefficient,audit


class ScoreControlTests(unittest.TestCase):
    def test_known_control(self):
        s=np.array([[-1,2],[1,-2]],float);g=np.array([3.,4.])+2.5*s
        self.assertAlmostEqual(fit_coefficient(g,s),2.5)
        self.assertEqual(fit_coefficient(g,np.zeros_like(s)),0)
        with self.assertRaises(ValueError):fit_coefficient(g,s,[-1,2])

    def test_exact_mean_variance_and_leakage(self):
        for p in [(-.15,.07),(0.,0.),(.2,-.1)]:
            r=audit(p);self.assertLessEqual(r['variance_after'],r['variance_before']+1e-12)
            self.assertGreater(r['same_sample_fit_bias_norm'],1e-8)


if __name__=='__main__':unittest.main()
