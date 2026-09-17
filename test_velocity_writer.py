import unittest
from types import SimpleNamespace
import numpy as np
from continuous_residual_pilot import correction,rollout
from train_closed_loop_writer import fit_velocity_scale
from train_closed_loop_policy import prefix_windows
from adaptive_search_prototype import AdaptiveBeam


class VelocityWriterTests(unittest.TestCase):
    def test_rest_oddness_and_bound(self):
        h=np.zeros((3,3,2));h[1,-1]=[.02,-.03];h[2]=-h[1]
        m=dict(kind='velocity',value=np.array([1e-4,-1e-4]),velocity_scale=np.array([.01,.01]))
        d=correction(m,None,h,None,None)
        np.testing.assert_array_equal(d[0],0)
        np.testing.assert_array_equal(d[1],-d[2]);self.assertTrue((np.abs(d)<=np.abs(m['value'])).all())
        np.testing.assert_allclose(correction(m,None,h+2,None,None),d,atol=1e-15)

    def test_scale_only_reads_current_and_past(self):
        y=np.arange(20,dtype=float).reshape(10,2);pool=SimpleNamespace(pool={12:dict(y=y,starts=np.array([2,3]))})
        a=fit_velocity_scale(pool);y[4:]+=10000
        np.testing.assert_array_equal(a,fit_velocity_scale(pool));np.testing.assert_array_equal(a,[2,2])

    def test_zero_replay_and_future_truth_isolation(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=50,per_video=1)
        m=dict(kind='velocity',value=np.zeros(2),velocity_scale=np.array([.1,.1]))
        _,p,f=rollout(s,w,None,29,particles=4);_,q,g=rollout(s,w,m,29,particles=4)
        np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(f,g)
        m['value']=np.array([1e-5,-2e-5])
        _,p,f=rollout(s,w,m,29,particles=4);_,q,g=rollout(s,dict(w,truth=w['truth']+2),m,29,particles=4)
        np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(f,g)


if __name__=='__main__':unittest.main()
