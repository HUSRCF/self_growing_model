import json
import unittest
import numpy as np
from soft_kernel_gate import scalar_optimum, objective, fit_gate, predict_gate


class SoftGateTests(unittest.TestCase):
    def test_scalar_nonconvex_and_endpoints(self):
        self.assertEqual(scalar_optimum([1.],[-2.]),1.)
        self.assertEqual(scalar_optimum([1.],[2.]),0.)
        self.assertAlmostEqual(scalar_optimum([-1.],[1.]),.5)
        self.assertEqual(scalar_optimum([0.],[0.]),0.)

    def test_gradient_with_saturated_states(self):
        z=np.array([-2.,0.,2.]); l=np.array([-.3,.2,-.1]); q=np.array([.2,-.1,.4])
        beta=np.array([.4,.7]); _,g=objective(beta,z,l,q)
        for j in range(2):
            d=np.eye(2)[j]*1e-6
            fd=(objective(beta+d,z,l,q)[0]-objective(beta-d,z,l,q)[0])/2e-6
            np.testing.assert_allclose(g[j],fd,atol=1e-9)

    def test_fit_only_normalization_and_serialization(self):
        x=np.linspace(-2,2,30); target=np.clip(.5+.2*x,0,1)
        model=fit_gate(x,-2*target,np.ones_like(x))
        self.assertTrue(model['success'])
        self.assertEqual(model['mean'],float(x.mean()))
        self.assertEqual(model['scale'],float(x.std()))
        saved=json.loads(json.dumps(model))
        np.testing.assert_array_equal(predict_gate(saved,x),predict_gate(model,x))
        points=np.array([-100.,.3,100.]); together=predict_gate(model,points)
        for i in range(3):
            self.assertEqual(predict_gate(model,points[i:i+1])[0],together[i])
        self.assertLessEqual(model['candidate_loss'],model['scalar_loss'])
