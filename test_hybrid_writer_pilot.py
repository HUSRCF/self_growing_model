import unittest
import numpy as np
import torch
from train_hybrid_writer import hybrid_loss
from ensemble_score_objective import energy_costs


class HybridPilotTests(unittest.TestCase):
    def test_objective_and_score_derivative(self):
        rng=np.random.default_rng(12)
        pred=torch.tensor(rng.normal(size=(4,300,2)),dtype=torch.float64,requires_grad=True)
        truth=torch.tensor(rng.normal(size=(300,2)),dtype=torch.float64)
        failed=torch.zeros((4,300),dtype=torch.bool);logs=torch.zeros(4,dtype=torch.float64,requires_grad=True)
        cost,loss=hybrid_loss(pred,truth,failed,logs);cs=[];bs=[]
        for t in [50,100,300]:
            x=pred.detach().numpy()[:,t-1];y=truth.numpy()[t-1]
            c,b=energy_costs(np.concatenate([np.sin(x),np.cos(x)],-1)[None],np.r_[np.sin(y),np.cos(y)][None],np.zeros((1,4),bool))
            cs.append(c[0]);bs.append(b[0])
        np.testing.assert_allclose(float(cost.detach()),np.mean(cs),atol=1e-14)
        np.testing.assert_allclose(torch.autograd.grad(loss,logs,retain_graph=True)[0],np.mean(cs)-np.mean(bs,axis=0),atol=1e-14)
        gp=torch.autograd.grad(cost,pred,retain_graph=True)[0];gh=torch.autograd.grad(loss,pred)[0]
        np.testing.assert_array_equal(gp,gh)


if __name__=='__main__':unittest.main()
