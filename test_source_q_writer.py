import unittest
import numpy as np
import torch
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import correction,rollout
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from train_closed_loop_policy import prefix_windows
from train_hybrid_writer import hybrid_loss


class SourceQWriterTests(unittest.TestCase):
    def test_tensor_indexes_source_and_routes_gradient(self):
        class Fake:
            def initialize(self,h):return torch.tensor([1,4]),torch.zeros((2,1),dtype=h.dtype)
            def read(self,h,q,hidden):
                return torch.ones((2,1),dtype=h.dtype),torch.nn.functional.one_hot((q+1)%8,8).to(h.dtype)[:,None],hidden
            def execute(self,h,q,r):return torch.zeros((2,2),dtype=h.dtype)
        table=torch.tensor(np.arange(16).reshape(8,2)/1000,dtype=torch.float64,requires_grad=True)
        a=sampled_path(Fake(),torch.zeros((2,32,2),dtype=torch.float64),1,table,351017)
        np.testing.assert_array_equal(a['prediction'].detach()[:,0],table.detach()[[1,4]])
        g=torch.autograd.grad(a['prediction'].sum(),table)[0]
        expected=np.zeros((8,2));expected[[1,4]]=1
        np.testing.assert_array_equal(g,expected)

    def test_indexes_source_not_destination(self):
        values=np.arange(16).reshape(8,2)
        actual=correction(dict(kind='source_q',value=values),None,None,np.array([1,4]),np.array([6,0]))
        np.testing.assert_array_equal(actual,values[[1,4]])

    def test_numpy_zero_and_shared_reduction(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=300,per_video=2)
        for value in [np.zeros(2),np.array([1e-6,-2e-6])]:
            shared=None if not value.any() else dict(kind='constant',value=value)
            a,p,f=rollout(s,w,shared,351017,particles=4)
            b,pp,ff=rollout(s,w,dict(kind='source_q',value=np.tile(value,(8,1))),351017,particles=4)
            self.assertEqual(a,b);np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(f,ff)

    def test_torch_shared_gradient_reduction(self):
        torch.set_num_threads(1);b=FrozenBridge(AdaptiveBeam());w=prefix_windows([12],steps=300,per_video=1)
        h=tensor(np.repeat(w['history'],4,axis=0));shared=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        table=torch.zeros((8,2),dtype=torch.float64,requires_grad=True);bound=tensor([4e-5,8e-5])
        a=sampled_path(b,h,300,shared*bound,351017,detach_every=50)
        aa=sampled_path(b,h,300,table*bound,351017,detach_every=50)
        for k in ['prediction','failed','logp','q','hidden']:np.testing.assert_array_equal(a[k].detach(),aa[k].detach())
        _,loss=hybrid_loss(a['prediction'],tensor(w['truth'][0]),a['failed'],a['logp'])
        _,other=hybrid_loss(aa['prediction'],tensor(w['truth'][0]),aa['failed'],aa['logp'])
        g=torch.autograd.grad(loss,shared)[0];gg=torch.autograd.grad(other,table)[0]
        np.testing.assert_allclose(gg.sum(0),g,rtol=1e-10,atol=1e-12)


if __name__=='__main__':unittest.main()
