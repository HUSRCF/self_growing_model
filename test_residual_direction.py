import unittest
import numpy as np
from audit_residual_direction import local_scores,state_snapshots


class ResidualDirectionTests(unittest.TestCase):
    def test_zero_correction_and_known_direction(self):
        c=np.array([[[.1,.2],[.2,.3]]]);y=np.zeros((1,2));p=np.array([[.4,.6]])
        zero=local_scores(c,y,p,np.zeros_like(c))
        np.testing.assert_array_equal(zero['mse_gain'],0);np.testing.assert_array_equal(zero['energy_gain'],0)
        good=local_scores(c,y,p,-c);self.assertGreater(good['mse_gain'][0],0)
        self.assertGreater(good['energy_gain'][0],0)
        eps=1e-6;small=local_scores(c,y,p,-c*eps)
        np.testing.assert_allclose(small['mse_gain'],small['first_order_mse_gain'],rtol=1e-5)

    def test_snapshots_match_zero_and_corrected_rollouts(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import prefix_windows
        from continuous_residual_pilot import rollout
        s=AdaptiveBeam();w=prefix_windows([12],steps=100,per_video=2)
        for model in [None,dict(kind='constant',value=np.array([1e-5,-1e-5]))]:
            # Match the exact three-particle row layout and RNG stream.
            expanded=np.repeat(w['history'],3,axis=0)
            snap=state_snapshots(s,expanded,1729,writer=model)
            _,pred,failed=rollout(s,w,model,1729,particles=3)
            for t in [0,8,24,100]:
                hist=np.concatenate([expanded,pred.reshape(6,100,2)[:,:t]],1)[:,-32:]
                np.testing.assert_array_equal(snap[t][0],hist)
                if t:np.testing.assert_array_equal(snap[t][3],failed.reshape(6,100)[:,t-1])


if __name__=='__main__':unittest.main()
