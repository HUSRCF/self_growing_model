import json
import unittest
import numpy as np
from residual_soft_gate import objective, predict, fit_gate, predict_gate, block_starts


class ResidualSoftGateTests(unittest.TestCase):
    def test_gradient(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(17, 9)); base = np.full(17, .5)
        l, q, beta = rng.normal(size=17), rng.normal(size=17), rng.normal(size=9)*.1
        _, gradient = objective(beta, x, base, l, q)
        eps = 1e-6
        numeric = []
        for i in range(9):
            step = np.eye(9)[i]*eps
            numeric.append((objective(beta+step,x,base,l,q)[0]-objective(beta-step,x,base,l,q)[0])/(2*eps))
        np.testing.assert_allclose(gradient, numeric, atol=1e-9, rtol=1e-6)

    def test_zero_bound_fallback_and_replay(self):
        x = np.ones((12, 9)); base = np.linspace(0, 1, 12)
        np.testing.assert_array_equal(predict(np.zeros(9), x, base), base)
        for sign in [-1, 1]:
            a = predict(np.full(9, sign*2.), x, base)
            self.assertTrue(((a>=0)&(a<=1)&(np.abs(a-base)<=.25)).all())
        g = fit_gate(x, base, np.zeros(12), np.zeros(12))
        self.assertFalse(g['use_residual'])
        np.testing.assert_array_equal(predict_gate(json.loads(json.dumps(g)), x, base), base)

    def test_temporal_purge(self):
        for length in [4321, 4556]:
            train, evaluation = block_starts(length,'train'), block_starts(length,'evaluation')
            self.assertEqual(len(set(train)), 4)
            self.assertTrue(train.min()-31 >= length//2)
            self.assertTrue(train.max()+300 < evaluation.min()-31)
            self.assertTrue(evaluation.max()+300 < length)
        with self.assertRaises(ValueError):
            block_starts(100, 'train')
