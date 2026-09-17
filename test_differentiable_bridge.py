import unittest
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
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

    def test_failure_keeps_sampling_and_shifting_history(self):
        class Fake:
            def initialize(self,h):return torch.zeros(len(h),dtype=torch.long),torch.zeros((len(h),1),dtype=h.dtype)
            def read(self,h,q,hidden):
                pe=torch.ones((len(h),1),dtype=h.dtype)
                tr=torch.nn.functional.one_hot(1-q,2).to(h.dtype)[:,None]
                return pe,tr,hidden+1
            def execute(self,h,q,r):return h[:,-1]+4
        h=tensor(np.arange(8).reshape(1,4,2));a=sampled_path(Fake(),h,3,tensor([0.,0.]),1,trace=True)
        np.testing.assert_array_equal(a['failed'],True)
        np.testing.assert_array_equal(a['prediction'],np.repeat(h[:,-1:].numpy(),3,axis=1))
        np.testing.assert_array_equal(a['q'],[1]);np.testing.assert_array_equal(a['hidden'],[[3.]])
        np.testing.assert_array_equal(a['history'],np.repeat(h[:,-1:].numpy(),4,axis=1))
        self.assertEqual([int(t['q'][0]) for t in a['trace']],[1,0,1])

    def test_sampled_path_retains_both_derivatives(self):
        b=FrozenBridge(AdaptiveBeam());h=tensor(prefix_windows([12],steps=10,per_video=1)['history'])
        theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        a=sampled_path(b,h,3,theta,291017)
        for v in [a['prediction'].sum(),a['logp'].sum()]:
            g=torch.autograd.grad(v,theta,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)


if __name__=='__main__':unittest.main()
