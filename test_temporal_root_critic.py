import json
import unittest
import numpy as np
from validate_temporal_root_critic import fit,score
from audit_value_coverage import predict_crossfit


class TemporalRootCriticTests(unittest.TestCase):
    def test_exact_original_fold_and_serialization(self):
        rng=np.random.default_rng(70)
        x=rng.normal(size=(7,3,5)); y=rng.normal(size=(7,3)); v=np.array([1,1,2,2,2,3,3])
        for contextual in [False,True]:
            use=v!=1; model=fit(x[use],y[use],v[use],contextual)
            expected=predict_crossfit(x,y,v,x[:2],v[:2],contextual)
            np.testing.assert_array_equal(score(model,x[:2]),expected)
            np.testing.assert_array_equal(score(model,x),score(json.loads(json.dumps(model)),x))
        model=fit(x,y,v,False)
        centered=y-y.mean(1,keepdims=True)
        np.testing.assert_array_equal(model['scores'],np.mean([centered[v==i].mean(0) for i in [1,2,3]],0))
