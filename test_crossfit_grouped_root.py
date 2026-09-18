import unittest
import numpy as np
from crossfit_grouped_root_constraint import grouped_probability


class CrossfitGroupedRootTests(unittest.TestCase):
    def test_causal_bin_and_zero_identity(self):
        p=np.full((3,8),.125);motion=np.array([1.,2.,3.]);video=np.array([1,1,2])
        direction=np.zeros((2,8));direction[0,2]=.25;direction[1,0]=.75
        m=dict(threshold=2.,direction_fit={'alpha':.2},local={'direction':direction.ravel().tolist()})
        folds={'1':m,'2':dict(threshold=0.,direction_fit={'alpha':0.},local=m['local'])}
        out=grouped_probability(folds,motion,video,p)
        np.testing.assert_allclose(out[0],.95*p[0]+.05*np.eye(8)[2])
        np.testing.assert_allclose(out[1],.85*p[1]+.15*np.eye(8)[0])
        np.testing.assert_array_equal(out[2],p[2])
