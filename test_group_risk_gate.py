import unittest
import numpy as np
from group_risk_gate import worst_scalar,risks,fit_gate
from soft_kernel_gate import scalar_optimum,predict_gate


class GroupRiskTests(unittest.TestCase):
    def test_scalar_global_against_grid_including_negative_curvature(self):
        rng=np.random.default_rng(311);grid=np.linspace(0,1,10001)
        for _ in range(30):
            l,q=rng.normal(size=2),rng.normal(size=2)
            a=worst_scalar(l,q,np.array([0,1]))
            self.assertLessEqual(np.max(l*a+q*a*a),np.max(l[:,None]*grid+q[:,None]*grid**2,axis=0).min()+1e-12)
        # Crossing, not either group's individual minimum, determines the optimum.
        self.assertAlmostEqual(worst_scalar(np.array([-1.,-.3]),np.array([1.,0.]),np.array([0,1])),.7)

    def test_group_gradients(self):
        z=np.linspace(-1,1,10);l=np.linspace(-.2,.1,10);q=np.linspace(-.1,.3,10);g=np.arange(10)%2
        beta=np.array([.5,.1]);eps=1e-6
        for mode in ['mean','worst']:
            _,grad=risks(beta,z,l,q,g,mode)
            finite=np.stack([(risks(beta+eps*v,z,l,q,g,mode)[0]-risks(beta-eps*v,z,l,q,g,mode)[0])/(2*eps) for v in np.eye(2)],1)
            np.testing.assert_allclose(grad,finite,atol=1e-9)

    def test_fallback_and_mean_scalar_control(self):
        x=np.linspace(-3,-1,20);g=np.arange(20)%2;l=np.full(20,.1);q=np.full(20,.2)
        worst=fit_gate(x,l,q,g)
        np.testing.assert_array_equal(predict_gate(worst,x),np.zeros(20))
        mean=fit_gate(x,l,q,g,'mean')
        self.assertEqual(mean['scalar'],scalar_optimum(l,q))
