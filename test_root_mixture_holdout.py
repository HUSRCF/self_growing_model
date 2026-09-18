import json
import unittest
import numpy as np
from audit_value_coverage import predict_crossfit
from validate_root_mixture_holdout import fit_critic, predict


class RootHoldoutTests(unittest.TestCase):
    def test_original_fit_reduction_and_serialization(self):
        rng = np.random.default_rng(831)
        x = rng.normal(size=(12, 8, 5)); y = rng.normal(size=(12, 8)); v = np.repeat([1, 2, 3], 4)
        xx = rng.normal(size=(3, 8, 5)); model = fit_critic(x, y)
        reference = predict_crossfit(x, y, v, xx, np.full(3, 99))
        np.testing.assert_allclose(predict(model, xx), reference, atol=1e-12)
        loaded = {k: np.array(v) for k, v in json.loads(json.dumps({k: v.tolist() for k, v in model.items()})).items()}
        np.testing.assert_array_equal(predict(model, xx), predict(loaded, xx))

    def test_prediction_batch_independence(self):
        rng = np.random.default_rng(86)
        x = rng.normal(size=(12, 8, 5)); y = rng.normal(size=(12, 8)); model = fit_critic(x, y)
        xx = rng.normal(size=(4, 8, 5)); all_scores = predict(model, xx)
        np.testing.assert_allclose(all_scores, np.concatenate([predict(model, xx[i:i+1]) for i in range(4)]), atol=1e-12)
