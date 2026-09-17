import unittest
import numpy as np
import torch
from audit_long_path_gradient import cost_and_adv,fixed_adv_objectives


class LongPathGradientTests(unittest.TestCase):
    def test_finite_difference_keeps_advantage_fixed(self):
        rng=np.random.default_rng(5);truth=torch.tensor(rng.normal(size=(300,2)),dtype=torch.float64)
        original=torch.tensor(rng.normal(size=(4,300,2)),dtype=torch.float64)
        adv=torch.stack([cost_and_adv(original,truth,t)[1] for t in [50,100,300]]).mean(0)
        theta=torch.tensor(.02,dtype=torch.float64,requires_grad=True)
        def f(t):return fixed_adv_objectives(original+t,torch.ones((4,300),dtype=torch.float64)*t,truth,adv)
        values=f(theta);g=np.array([float(torch.autograd.grad(v,theta,retain_graph=True)[0]) for v in values])
        fd=((f(theta.detach()+1e-6)-f(theta.detach()-1e-6))/(2e-6)).numpy()
        np.testing.assert_allclose(fd,g,rtol=1e-6,atol=1e-8)
        self.assertFalse(adv.requires_grad)


if __name__=='__main__':unittest.main()
