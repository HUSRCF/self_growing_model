import unittest
import numpy as np
from audit_root_video_feasibility import positive_interval,fit_robust


class RootVideoFeasibilityTests(unittest.TestCase):
    def test_intervals(self):
        self.assertIsNone(positive_interval([.1],[1.]))
        self.assertEqual(positive_interval([.1],[-1.]),[.1,.25])
        self.assertEqual(positive_interval([.1,-.2],[-1.,1.]),[.1,.2])
        self.assertIsNone(positive_interval([.2,-.1],[-1.,1.]))
        self.assertEqual(positive_interval([0.],[0.]),[0.,.25])
        self.assertIsNone(positive_interval([1.],[0.]))
        with self.assertRaises(ValueError):positive_interval([np.nan],[1.])

    def test_random_dense_grid(self):
        rng=np.random.default_rng(751);grid=np.linspace(0,.25,10001)
        for _ in range(50):
            l,q=rng.normal(size=(2,4,3));result=fit_robust(l,q)
            d=grid[:,None,None]*l+grid[:,None,None]**2*q
            valid=(d[:,:,0]<=1e-12).all(1)
            self.assertLessEqual(result['mean_delta'],d[valid].mean((1,2)).min()+1e-12)
