import unittest
import numpy as np
from first_point_distribution import draw_noise
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows


class FirstPointTests(unittest.TestCase):
    def test_noise_zero_and_covariance(self):
        np.testing.assert_array_equal(draw_noise(np.zeros((2,2)),20,3),np.zeros((20,2)))
        c=np.array([[2.,.3],[.3,1.]])
        np.testing.assert_allclose(np.cov(draw_noise(c,100000,3).T),c,atol=.025)

    def test_first_only_and_zero_parity(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=50,per_video=1)
        _,a,f=rollout(s,w,None,3,particles=4)
        _,b,g=rollout(s,w,None,3,particles=4,first_noise=np.zeros((4,2)))
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(f,g)
        fixed=w['history'][0,-1].copy()
        s.base.execute_rule=lambda h,q,r:np.broadcast_to(fixed,(len(h),2)).copy()
        noise=np.full((4,2),.0001)
        _,p,_=rollout(s,w,None,3,particles=4,first_noise=noise)
        np.testing.assert_allclose(p[0,:,0],fixed[None]+noise)
        np.testing.assert_allclose(p[0,:,1:],np.broadcast_to(fixed,p[0,:,1:].shape))
        with self.assertRaises(ValueError):rollout(s,w,None,3,particles=4,first_noise=np.zeros((1,2)))


if __name__=='__main__':unittest.main()
