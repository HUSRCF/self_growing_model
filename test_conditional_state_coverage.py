import unittest
import numpy as np
from audit_conditional_state_coverage import summarize


class CoverageTests(unittest.TestCase):
    def test_conditional_prior_and_predicted_gap(self):
        y=np.array([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]])
        rows=[dict(labels=y.tolist()),dict(labels=(y+2).tolist())]
        prior=np.array([[.25,.75],[.5,.5]])
        choices={'context':np.array([0,1])};scores={'context':np.array([[-1.,1.],[1.,-1.]])}
        result=summarize(rows,prior,choices,scores,np.array([1,2]))['context']
        self.assertAlmostEqual(result['delta'],-.25)
        self.assertAlmostEqual(result['predicted_delta'],-1.25)
        self.assertEqual(result['per_video_delta'],{'1':-1.5,'2':1.})
        np.testing.assert_array_equal(result['horizon_delta'],[-.25,-.25])


if __name__=='__main__':unittest.main()
