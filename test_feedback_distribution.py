import unittest
import numpy as np
from feedback_distribution_pilot import corrected_tr, fit_covariance, gaussian_audit, sample


class PilotTests(unittest.TestCase):
    def test_zero_temporal_correlation_matches_independent_sampling(self):
        from adaptive_search_prototype import AdaptiveBeam
        from feedback_distribution_pilot import rollout_arm
        from v20_rnn_mixture.engine.data import tail_windows
        s=AdaptiveBeam();windows=tail_windows('dev',300,1)
        windows['truth']=windows['truth'][:,:10]
        cov=np.tile(np.eye(2)*1e-8,(8,8,1,1))
        _,p,f=rollout_arm(s,windows,None,cov,False,True,1729,particles=2)
        _,p0,f0=rollout_arm(s,windows,None,cov,False,True,1729,particles=2,rho=np.zeros(2))
        np.testing.assert_array_equal(p,p0)
        np.testing.assert_array_equal(f,f0)

    def test_residual_transition_preserves_event_conditional_normalization(self):
        rng=np.random.default_rng(1)
        tr=rng.random((4,8,8));tr/=tr.sum(-1,keepdims=True)
        x=rng.normal(size=(4,3));fb=rng.normal(size=(4,7))
        a=dict(mean=np.zeros(10),scale=np.ones(10),w1=np.ones((10,2)),
               b1=np.zeros(2),w2=np.zeros((2,8)),b2=np.zeros(8))
        np.testing.assert_allclose(corrected_tr(a,x,fb,tr,True),tr)
        a['w2'][0,0]=2
        result=corrected_tr(a,x,fb,tr,True)
        np.testing.assert_allclose(result.sum(-1),1)
        self.assertTrue((result>0).all())
        np.testing.assert_array_equal(corrected_tr(a,x,fb,tr,False),
                                      corrected_tr(a,x,fb*7,tr,False))
        self.assertFalse(np.allclose(result,corrected_tr(a,x,np.zeros_like(fb),tr,True)))

    def test_covariance_stays_about_fixed_center_and_uses_fit_videos_only(self):
        data=dict(residual=np.array([[1.,2.],[1.,2.],[1e9,1e9]]),
                  q=np.array([0,0,0]),r=np.array([1,1,1]),video=np.array([12,12,18]))
        cov,counts=fit_covariance(data)
        self.assertEqual(counts.sum(),2)
        # Constant nonzero residual must not disappear through mean subtraction.
        self.assertGreater(cov[0,1,0,0],.99)
        self.assertLess(cov[0,1,0,0],1.01)
        self.assertTrue((np.linalg.eigvalsh(cov)>0).all())
        audit=gaussian_audit({k:v[:2] for k,v in data.items()},cov)
        self.assertTrue(np.isfinite(audit['nll']))

    def test_event_sampling_never_leaves_alphabet(self):
        probs=np.eye(8)
        np.testing.assert_array_equal(sample(probs,np.full(8,.999999)),np.arange(8))


if __name__=='__main__':unittest.main()
