import unittest
import numpy as np
from train_matched_state_coverage import datasets


class MatchedCoverageTests(unittest.TestCase):
    def test_budget_and_group_alignment(self):
        x=np.arange(4*3*2*5).reshape(4,3,2,5)
        y=np.arange(4*4*3*2*2).reshape(4,4,3,2,2)
        v=np.array([1,2,3]);d=datasets(x,y,v)
        np.testing.assert_array_equal(d['single'][0],x[0])
        np.testing.assert_array_equal(d['single'][1],y[0].mean((0,3)))
        for g in range(4):
            np.testing.assert_array_equal(d['multi'][0][3*g:3*g+3],x[g])
            np.testing.assert_array_equal(d['multi'][1][3*g:3*g+3],y[g,g].mean(-1))
            np.testing.assert_array_equal(d['multi'][2][3*g:3*g+3],v)
        # 4 futures x 4 branches at one state vs 4 states x 4 branches.
        self.assertEqual(len(d['single'][0])*16,len(d['multi'][0])*4)
        altered=y.copy();altered[3,0]+=10000
        other=datasets(x,altered,v)
        np.testing.assert_array_equal(other['single'][1],d['single'][1])
        np.testing.assert_array_equal(other['multi'][1],d['multi'][1])
        with self.assertRaises(ValueError):datasets(x[:3],y,v)


if __name__=='__main__':unittest.main()
