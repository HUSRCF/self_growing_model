import unittest
import numpy as np
from train_closed_loop_writer import candidates,constant_model
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from adaptive_search_prototype import AdaptiveBeam


class ClosedLoopWriterTests(unittest.TestCase):
    def test_search_box_and_incumbent(self):
        theta=np.array([.99,-.99]);p=candidates(theta,np.array([1.,-1.]),1)
        np.testing.assert_array_equal(p[0],theta);self.assertTrue((np.abs(p)<=1).all())
        self.assertLessEqual(np.linalg.norm(candidates(np.zeros(2),[1,2],12)[1]),.5/np.sqrt(12)+1e-15)
        with self.assertRaises(ValueError):constant_model(np.array([2.,0]),np.ones(2))

    def test_zero_exact_and_truth_isolation(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=50,per_video=1);bound=np.array([4e-5,8e-5])
        _,p,f=rollout(s,w,None,19,particles=4)
        _,z,b=rollout(s,w,constant_model(np.zeros(2),bound),19,particles=4)
        np.testing.assert_array_equal(p,z);np.testing.assert_array_equal(f,b)
        model=constant_model(np.array([.2,-.3]),bound)
        _,p,f=rollout(s,w,model,19,particles=4)
        _,z,b=rollout(s,dict(w,truth=w['truth']+1),model,19,particles=4)
        np.testing.assert_array_equal(p,z);np.testing.assert_array_equal(f,b)


if __name__=='__main__':unittest.main()
