import unittest
import numpy as np
from crossfit_root_constraint import fit_folds,select,probabilities


class CrossfitRootConstraintTests(unittest.TestCase):
    def test_excluded_video_labels_and_normalization(self):
        rng=np.random.default_rng(731)
        l=rng.normal(size=(6,8,3));q=rng.normal(size=l.shape);v=np.repeat([1,2,3],2)
        folds=fit_folds(l,q,v)
        altered=l.copy();altered[:2]+=100*rng.normal(size=(2,8,3))
        self.assertEqual(folds['1'],fit_folds(altered,q,v)['1'])
        p=np.full((6,8),.125)
        for value in probabilities(folds,v,p).values():
            np.testing.assert_allclose(value.sum(1),1.)
            self.assertTrue((value>=0).all())
        for f in folds.values():self.assertLessEqual(f['constrained']['short_violation'],1e-12)

    def test_prior_when_all_changes_bad(self):
        selected=select(np.ones((8,3)),np.ones((8,3)),True)
        self.assertEqual(selected['alpha'],0.)
        self.assertEqual(selected['mean_delta'],0.)
