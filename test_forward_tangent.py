import unittest
import numpy as np
import torch
from audit_forward_tangent import propagate


class ForwardTangentTests(unittest.TestCase):
    def test_known_recurrence_and_block_sum(self):
        class Linear:
            def initialize(self,h):return torch.zeros(len(h),dtype=torch.long),torch.zeros((len(h),1),dtype=h.dtype)
            def read(self,h,q,hidden):return torch.ones((len(h),1),dtype=h.dtype),torch.ones((len(h),1,1),dtype=h.dtype),hidden+h[:,-1,:1]
            def execute(self,h,q,r):return 2*h[:,-1]
        h=torch.ones((3,2,2),dtype=torch.float64)*.1;rr=torch.zeros((3,3),dtype=torch.long)
        bound=torch.ones(2,dtype=torch.float64);direction=torch.tensor([1.,0.],dtype=torch.float64)
        truth=torch.zeros((3,2),dtype=torch.float64);adv=torch.zeros(3,dtype=torch.float64)
        args=(Linear(),h,rr,bound,direction,truth,adv)
        _,g,history=propagate(*args,horizons=(1,2,3))
        np.testing.assert_array_equal(history[-1]['position_tangent'],np.tile([7.,0.],(3,1)))
        parts=[propagate(*args,block=(i,i+1),horizons=(1,2,3))[1] for i in range(3)]
        np.testing.assert_allclose(sum(parts),g,atol=1e-14)


if __name__=='__main__':unittest.main()
