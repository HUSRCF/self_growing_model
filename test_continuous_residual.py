import unittest
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import prefix_windows,run_policy
from continuous_residual_pilot import fit_models,correction,rollout


class ContinuousResidualTests(unittest.TestCase):
    def setUp(self):
        self.s=AdaptiveBeam()

    def test_holdout_changes_do_not_change_fit_parameters(self):
        w=prefix_windows([12,18],steps=1,per_video=2);h=w['history'];q=self.s.base.state_from_history(h)[0]
        data=dict(history=h,q=q,truth=w['truth'][:,0],probability=np.full((len(h),8),1/8),video=w['video'])
        models,_,_=fit_models(self.s.base,data)
        changed={k:v.copy() for k,v in data.items()};hold=changed['video']==18
        changed['history'][hold]+=.1;changed['truth'][hold]+=1
        other,_,_=fit_models(self.s.base,changed)
        for name,m in models.items():
            if m is not None:
                for k,v in m.items():np.testing.assert_array_equal(v,other[name][k])

    def test_linear_correction_is_bounded(self):
        w=prefix_windows([12],steps=1,per_video=2);h=w['history'];q=self.s.base.state_from_history(h)[0]
        models,_,_=fit_models(self.s.base,dict(history=h,q=q,truth=w['truth'][:,0],probability=np.full((len(h),8),1/8),video=w['video']))
        m=models['ridge_0.01'];m['coef']=m['coef']*1e10
        delta=correction(m,self.s.base,h,q,q)
        self.assertTrue((np.abs(delta)<=m['cap']+1e-15).all())

    def test_zero_writer_is_exact_original(self):
        w=prefix_windows([12],steps=100,per_video=1)
        _,p,f=rollout(self.s,w,None,1729,particles=4)
        _,ref,bad=run_policy(self.s,w,seed=1729,particles=4)
        np.testing.assert_array_equal(p,ref);np.testing.assert_array_equal(f,bad)

    def test_correction_inference_does_not_read_future_truth(self):
        w=prefix_windows([12],steps=50,per_video=1);model=dict(kind='constant',value=np.array([1e-5,-1e-5]))
        _,p,f=rollout(self.s,w,model,1729,particles=4)
        _,q,g=rollout(self.s,dict(w,truth=w['truth']+1),model,1729,particles=4)
        np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(f,g)


if __name__=='__main__':unittest.main()
