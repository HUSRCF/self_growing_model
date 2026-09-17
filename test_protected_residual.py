import unittest
import numpy as np
from protected_residual_pilot import protected_scale,scale_model


class ProtectedResidualTests(unittest.TestCase):
    def test_analytic_optimum_and_hold_isolation(self):
        d=np.ones((2,2,2));target=d*.3;p=np.full((2,2),.5);fit=np.array([True,False])
        a,z=protected_scale(target,d,p,fit);self.assertAlmostEqual(a,.3)
        self.assertLess(z['quadratic_change'],0)
        target[1]=1e9;d[1]=1e8;p[1]=1e7
        b,other=protected_scale(target,d,p,fit);self.assertEqual(a,b);self.assertEqual(z,other)
        self.assertEqual(protected_scale(-d,d,p,fit)[0],0)
        self.assertEqual(protected_scale(2*d,d,p,fit)[0],1)
        self.assertEqual(protected_scale(d,np.zeros_like(d),p,fit)[0],0)

    def test_scaled_cap_preserves_bounded_direction(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(20,3))
        m=dict(coef=rng.normal(size=(3,2)),cap=np.array([.2,.1]))
        original=m['coef'].copy()
        for a in [0,.01,1]:
            other=scale_model(m,a)
            np.testing.assert_allclose(np.clip(x@other['coef'],-other['cap'],other['cap']),
                                       a*np.clip(x@m['coef'],-m['cap'],m['cap']),atol=1e-15)
        np.testing.assert_array_equal(m['coef'],original)


if __name__=='__main__':unittest.main()
