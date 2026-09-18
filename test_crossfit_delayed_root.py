import unittest
import numpy as np
from crossfit_delayed_root import fit_folds,policies


class CrossfitDelayedRootTests(unittest.TestCase):
    def test_excluded_video_labels_and_component_index(self):
        rng=np.random.default_rng(781);l=rng.normal(size=(6,8,3));q=rng.normal(size=l.shape);l[:,:,0]=0;q[:,:,0]=0
        v=np.repeat([1,2,3],2);f=fit_folds(l,q,v)
        alt=l.copy();alt[:2]+=100
        self.assertEqual(f['1'],fit_folds(alt,q,v)['1'])
        p=policies(f,v)
        np.testing.assert_array_equal(p['zero'][:,0],1.)
        np.testing.assert_array_equal(p['fixed6'][:,7],.25)
        for value in p.values():
            np.testing.assert_allclose(value.sum(1),1.);self.assertTrue((value>=0).all())

    def test_bad_directions_select_zero(self):
        l=np.ones((4,8,3));q=np.ones_like(l);l[:,:,0]=0;q[:,:,0]=0;v=np.array([1,1,2,2])
        f=fit_folds(l,q,v)
        np.testing.assert_array_equal(policies(f,v)['crossfit'],policies(f,v)['zero'])
