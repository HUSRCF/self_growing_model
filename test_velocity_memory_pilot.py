import unittest
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import rollout
from train_closed_loop_policy import prefix_windows
from velocity_memory_compat import LegacyFeatureBridge
from velocity_memory_pilot import prefix_sequence,initializer_data,fit_initializer,rollout_memory


class VelocityMemoryPilotTests(unittest.TestCase):
    def test_prefix_truncated_before_centered_labels_and_fit(self):
        s=AdaptiveBeam();base=LegacyFeatureBridge(s.base)
        t=np.arange(160);full=np.stack([np.sin(t*.03),np.cos(t*.07)],axis=1)
        changed=full.copy();changed[80:]+=1e5
        a=[(12,prefix_sequence(full))];b=[(12,prefix_sequence(changed))]
        for x,y in zip(initializer_data(base,a),initializer_data(base,b)):np.testing.assert_array_equal(x,y)
        ma,_=fit_initializer(base,a);mb,_=fit_initializer(base,b)
        for key in ma:np.testing.assert_array_equal(ma[key],mb[key])

    def test_frozen_writer_replay_and_future_truth_isolation(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=50,per_video=1)
        class FrozenWriter:
            def initialize(self,h):return np.zeros((len(h),2))
            def execute(self,h,q,r,v):return s.base.execute_rule(h,q,r),v.copy()
        writer=FrozenWriter();_,p,f,_=rollout_memory(s,w,writer,31,particles=4)
        _,old,bad=rollout(s,w,None,31,particles=4)
        np.testing.assert_array_equal(p,old);np.testing.assert_array_equal(f,bad)
        _,other,failed,_=rollout_memory(s,dict(w,truth=w['truth']+1),writer,31,particles=4)
        np.testing.assert_array_equal(p,other);np.testing.assert_array_equal(f,failed)

    def test_failure_freezes_all_committed_memory(self):
        s=AdaptiveBeam();w=prefix_windows([12],steps=3,per_video=1)
        class FailingWriter:
            count=0
            def initialize(self,h):return np.full((len(h),2),.001)
            def execute(self,h,q,r,v):
                self.count+=1
                if self.count==1:return h[:,-1]+v,v.copy()
                return np.full_like(v,np.nan),np.full_like(v,np.nan)
        _,p,f,records=rollout_memory(s,w,FailingWriter(),31,particles=4,trace=True)
        self.assertFalse(f[:,:,0].any());self.assertTrue(f[:,:,1:].all())
        for record in records[1:]:
            for key in ['history','hidden','velocity','destination']:
                np.testing.assert_array_equal(record[key],records[0][key])
        np.testing.assert_array_equal(p[:,:,1],p[:,:,0])


if __name__=='__main__':unittest.main()
