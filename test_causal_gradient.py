import unittest
import numpy as np
import torch
from audit_causal_gradient import toy
from train_hybrid_writer import hybrid_loss


class CausalGradientTests(unittest.TestCase):
    def test_exact_expectation(self):
        for p in [(-.15,.07),(0.,0.),(.2,-.1)]:toy(p)

    def test_prefix_cannot_use_future_logs(self):
        rng=np.random.default_rng(1)
        pred=torch.tensor(rng.normal(size=(4,300,2)),dtype=torch.float64)
        truth=torch.tensor(rng.normal(size=(300,2)),dtype=torch.float64)
        increments=torch.zeros((4,300),dtype=torch.float64,requires_grad=True)
        logs=increments.cumsum(1);failed=torch.zeros((4,300),dtype=torch.bool)
        _,loss=hybrid_loss(pred,truth,failed,logs[:,-1],logs)
        g=torch.autograd.grad(loss,increments)[0]
        for start,stop in [(0,50),(50,100),(100,300)]:
            np.testing.assert_allclose(g[:,start:stop],g[:,start:start+1].expand(-1,stop-start),atol=1e-14)
        self.assertGreater(float((g[:,49]-g[:,50]).abs().max()),1e-8)


if __name__=='__main__':unittest.main()
