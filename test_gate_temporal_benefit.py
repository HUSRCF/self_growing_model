import unittest
from unittest.mock import patch
import numpy as np
from audit_gate_temporal_benefit import score_chunk


class TemporalBenefitTests(unittest.TestCase):
    def test_chunk_preserves_alignment_and_scores_all_weights(self):
        n = 40
        row = dict(region='tail', seed=611017, motion=np.arange(n),
                   points=np.zeros((n, 3, 3, 2)), truth=np.zeros((n, 3, 2)),
                   failed=np.zeros((n, 3, 3), bool),
                   alpha={'zero': np.zeros((n, 3)), 'full': np.ones((n, 3)),
                          'features': np.arange(n*3).reshape(n, 3)/(n*3)})
        baseline = np.ones((20, 3))
        terms = dict(baseline=baseline.tolist(), linear=(-2*baseline).tolist(), quadratic=baseline.tolist())
        precision = dict(cost=0., gradient=0., cost_pass=True, gradient_pass=True)
        with patch('audit_gate_temporal_benefit.coefficients', return_value=({'4097': terms}, precision)) as mock:
            result = score_chunk({}, row, 20)
        np.testing.assert_array_equal(mock.call_args.args[1], row['motion'][20:])
        self.assertEqual(result['begin'], 20)
        self.assertEqual(result['region'], 'tail')
        np.testing.assert_array_equal(result['costs']['zero'], baseline)
        np.testing.assert_array_equal(result['costs']['full'], np.zeros((20, 3)))
        np.testing.assert_allclose(result['costs']['features'], (1-row['alpha']['features'][20:])**2)
