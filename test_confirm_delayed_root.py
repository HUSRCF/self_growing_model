import unittest
import numpy as np
from confirm_delayed_root import score_pair
from audit_full_root_distribution import mixture_terms,mixture_score


class ConfirmDelayedRootTests(unittest.TestCase):
    def test_two_components_equal_nine_with_zero_weights(self):
        rng=np.random.default_rng(781);x=rng.normal(size=(2,9,4,4));y=rng.normal(size=(2,4));f=rng.random((2,9,4))<.1
        a,b=mixture_terms(x,y,f);prior=np.zeros((2,9));prior[:,0]=1;mix=.75*prior;mix[:,7]=.25
        z,m=score_pair(x[:,[0,7]],y,f[:,[0,7]])
        np.testing.assert_allclose(z,mixture_score(a,b,prior),atol=1e-14,rtol=0)
        np.testing.assert_allclose(m,mixture_score(a,b,mix),atol=1e-14,rtol=0)
