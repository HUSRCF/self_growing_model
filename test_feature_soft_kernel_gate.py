import json
import unittest
import numpy as np
from feature_soft_kernel_gate import fit_mapping,design,objective,fit_gate,predict_gate


class FeatureGateTests(unittest.TestCase):
    def test_mapping_fit_only_and_batch_independence(self):
        raw=np.random.default_rng(71).normal(size=(20,14))
        mapping=fit_mapping(raw[:12]);saved=json.loads(json.dumps(mapping))
        np.testing.assert_array_equal(mapping['mean'],raw[:12].mean(0))
        before=design(saved,raw[12:]);raw[13:]*=1000
        # GEMV/GEMM reduction order can differ at float64 roundoff across batch sizes.
        np.testing.assert_allclose(before[:1],design(saved,raw[12:13]),atol=1e-14,rtol=0)
        np.testing.assert_allclose(before[:1],design(saved,raw[12:])[:1],atol=1e-14)

    def test_gradient_and_fit(self):
        rng=np.random.default_rng(21);raw=rng.normal(size=(40,11));x=design(fit_mapping(raw),raw)
        target=np.clip(.5+.3*x[:,1],0,1);l=-2*target;q=np.ones(40)
        beta=np.r_[.4,np.full(8,.03)];_,g=objective(beta,x,l,q)
        for j in range(len(beta)):
            d=np.eye(len(beta))[j]*1e-6
            fd=(objective(beta+d,x,l,q)[0]-objective(beta-d,x,l,q)[0])/2e-6
            np.testing.assert_allclose(g[j],fd,atol=1e-9)
        model=fit_gate(x,l,q);self.assertTrue(model['success'])
        np.testing.assert_array_equal(predict_gate(model,x),predict_gate(json.loads(json.dumps(model)),x))
        self.assertLessEqual(model['candidate_loss'],model['scalar_loss'])
