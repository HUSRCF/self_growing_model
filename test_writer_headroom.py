import unittest
import numpy as np
from audit_writer_headroom import decomposition,free_snapshots,inspect


class WriterHeadroomTests(unittest.TestCase):
    def test_variance_identity_and_oracle_is_not_ensemble_bound(self):
        centers=np.array([[[-.2,0],[.2,0]]]);truth=np.zeros((1,2));p=np.array([[.5,.5]])
        result=decomposition(centers,truth,p)
        np.testing.assert_allclose(result['expected_mse'],result['ensemble_mse']+result['candidate_variance'])
        self.assertGreater(result['oracle_mse'][0],result['ensemble_mse'][0])
        other=decomposition(centers,truth+.1,p)
        np.testing.assert_array_equal(result['candidate_variance'],other['candidate_variance'])

    def test_identical_candidates_have_no_routing_headroom(self):
        centers=np.zeros((2,3,2));p=np.array([[.1,.2,.7],[.8,.1,.1]])
        result=decomposition(centers,np.ones((2,2)),p)
        np.testing.assert_allclose(result['routing_headroom'],0,atol=1e-14)
        np.testing.assert_allclose(result['candidate_variance'],0,atol=1e-14)

    def test_free_histories_match_original_frozen_rollout(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import prefix_windows,run_policy
        s=AdaptiveBeam();w=prefix_windows([12],steps=100,per_video=2)
        snapshots=free_snapshots(s,w['history'],1729)
        _,pred,failed=run_policy(s,w,particles=1,seed=1729)
        for t in [0,24,100]:
            history=np.concatenate([w['history'],pred[:,0,:t]],axis=1)[:,-32:]
            np.testing.assert_array_equal(snapshots[t][0],history)
            if t:np.testing.assert_array_equal(snapshots[t][3],failed[:,0,t-1])

    def test_supported_restrictions_cannot_improve_point_oracle(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import prefix_windows
        s=AdaptiveBeam();w=prefix_windows([12],steps=1,per_video=2)
        h=w['history'];q,mem=s.machine.initialize(h);before=h.copy();hidden=mem['hidden'].copy()
        result,_,_,_=inspect(s,h,q,mem['hidden'],w['truth'][:,0])
        for key in ['current_check_oracle_mse','consistent_oracle_mse']:
            valid=np.isfinite(result[key])
            self.assertTrue((result[key][valid]>=result['oracle_mse'][valid]-1e-14).all())
        np.testing.assert_array_equal(h,before);np.testing.assert_array_equal(mem['hidden'],hidden)


if __name__=='__main__':unittest.main()
