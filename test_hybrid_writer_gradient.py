import unittest
import numpy as np
import torch
from hybrid_writer_gradient import audit,paths,objectives


class HybridGradientTests(unittest.TestCase):
    def test_exact_hybrid_and_finite_difference(self):
        for point in [(-.15,.07),(0.,0.),(.2,-.1)]:
            r=audit(point)
            # Test nonzero omission, not an assumed large practical effect.
            self.assertGreater(r['pathwise_bias_norm'],1e-8)
            self.assertGreater(r['score_only_bias_norm'],1e-4)

    def test_detached_routing_hides_dependency(self):
        theta=torch.tensor([.1,.05],dtype=torch.float64,requires_grad=True)
        x,lp=paths(theta);xx,detached=paths(theta,True)
        np.testing.assert_array_equal(x.detach(),xx.detach())
        np.testing.assert_array_equal(lp.detach(),detached.detach())
        self.assertTrue(lp.requires_grad);self.assertFalse(detached.requires_grad)
        self.assertGreater(float(torch.linalg.vector_norm(torch.autograd.grad(lp[0],theta)[0])),0.)

    def test_probability_normalization_gradient(self):
        theta=torch.tensor([.1,.05],dtype=torch.float64,requires_grad=True)
        value=objectives(theta)['probability_sum']
        np.testing.assert_allclose(torch.autograd.grad(value,theta)[0],0,atol=1e-14)


if __name__=='__main__':unittest.main()
