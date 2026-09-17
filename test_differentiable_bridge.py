import unittest
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor
from train_closed_loop_policy import prefix_windows


class BridgeTests(unittest.TestCase):
    def test_forward_and_frozen_weights(self):
        s=AdaptiveBeam();b=FrozenBridge(s);h=prefix_windows([12],steps=10,per_video=2)['history'];th=tensor(h)
        q,m=s.machine.initialize(h);tq,tm=b.initialize(th)
        np.testing.assert_array_equal(tq,q);np.testing.assert_allclose(tm,m['hidden'],atol=1e-11)
        r=(q+1)%8;actual=b.execute(th,tq,torch.tensor(r))
        np.testing.assert_allclose(actual,s.base.execute_rule(h,q,r),atol=1e-11)
        self.assertTrue(all(not v.requires_grad for v in b.head.values()))
        self.assertTrue(all(not v.requires_grad for v in b.weights))

    def test_constant_context_gradient_finite(self):
        b=FrozenBridge(AdaptiveBeam());h=torch.ones((2,32,2),dtype=torch.float64,requires_grad=True)
        v=b.context(h).sum();g=torch.autograd.grad(v,h)[0]
        self.assertTrue(torch.isfinite(g).all())
        np.testing.assert_allclose(b.context(h).detach(),0,atol=1e-15)


if __name__=='__main__':unittest.main()
