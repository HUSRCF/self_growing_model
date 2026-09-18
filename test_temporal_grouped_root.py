import json
import unittest
import numpy as np
from validate_temporal_grouped_root import full_probabilities


class TemporalGroupedRootTests(unittest.TestCase):
    def test_full_models_serialization_and_controls(self):
        direction=np.zeros((2,8));direction[:,0]=.5
        models=dict(grouped=dict(threshold=2.,direction_fit={'alpha':.4},local={'direction':direction.ravel().tolist()}),
                    constrained={'root':5,'alpha':.243},unconstrained={'root':7,'alpha':.25})
        prior=np.full((2,8),.125);motion=np.array([1.,3.])
        result=full_probabilities(models,motion,prior)
        np.testing.assert_array_equal(result['zero'],prior)
        np.testing.assert_allclose(result['grouped'],.8*prior+.2*np.eye(8)[0])
        np.testing.assert_allclose(result['constrained'],.757*prior+.243*np.eye(8)[5])
        for k,p in full_probabilities(json.loads(json.dumps(models)),motion,prior).items():
            np.testing.assert_array_equal(result[k],p)
            np.testing.assert_allclose(p.sum(1),1.)
