import unittest
import numpy as np
from crossfit_root_mixture_critic import targets, policy_features
from audit_value_coverage import predict_crossfit
from audit_full_root_distribution import change_terms


class RootMixtureCriticTests(unittest.TestCase):
    def test_target_exact_decomposition(self):
        p = np.array([[.2,.8]])
        a = np.array([[1.,2.]])
        b = np.array([[[0.,1.],[1.,0.]]])
        linear, full, policies = targets(a,b,p)
        self.assertEqual(full[0,0], 0.)
        for i in range(3):
            l,q = change_terms(a,b,policies[:,i],p)
            np.testing.assert_allclose(full[:,i],l+q,atol=1e-15)
            np.testing.assert_array_equal(linear[:,i],l)

    def test_policy_features_and_excluded_labels(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(6,2,3)); p = np.full((6,2),.5)
        xx = policy_features(x,p)
        np.testing.assert_allclose(xx[:,0],x.mean(1))
        y = rng.normal(size=(6,3)); v = np.repeat([1,2,3],2)
        for context in [False,True]:
            first = predict_crossfit(xx,y,v,xx[:2],v[:2],context)
            altered = y.copy(); altered[:2] += 100*rng.normal(size=(2,3))
            second = predict_crossfit(xx,altered,v,xx[:2],v[:2],context)
            np.testing.assert_array_equal(first,second)
