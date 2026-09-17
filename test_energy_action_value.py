import unittest
import numpy as np
from ensemble_score_objective import energy_costs
from audit_energy_action_value import replacement_delta, action_labels, embedding


class EnergyActionTests(unittest.TestCase):
    def test_direct_replacement_and_zero(self):
        rng = np.random.default_rng(811)
        for p in (3, 4, 8):
            x = rng.normal(size=(5, p, 4)); y = rng.normal(size=(5, 4))
            f = rng.random((5, p)) < .3
            original = energy_costs(x, y, f)[0]
            for i in range(p):
                c = rng.normal(size=(5, 4)); cf = rng.random(5) < .3
                z = x.copy(); z[:, i] = c
                ff = f.copy(); ff[:, i] = cf
                np.testing.assert_allclose(original+replacement_delta(x, y, f, c, cf, i), energy_costs(z, y, ff)[0], atol=1e-14)
                np.testing.assert_array_equal(replacement_delta(x, y, f, x[:, i], f[:, i], i), 0)

    def test_pair_term_and_failure_weight(self):
        x = np.zeros((1, 4, 1)); y = np.zeros((1, 1)); f = np.zeros((1, 4), bool)
        # Truth attraction and diversity cancel when moving one coincident particle.
        np.testing.assert_allclose(replacement_delta(x, y, f, np.ones((1, 1)), np.array([False]), 0), 0)
        np.testing.assert_allclose(replacement_delta(x, y, f, np.ones((1, 1)), np.array([True]), 0), .5)

    def test_single_particle_is_not_simultaneous_policy_change(self):
        base = np.zeros((1, 4, 300, 2)); candidate = np.full_like(base, np.pi)
        failed = np.zeros((1, 4, 300), bool); truth = np.zeros((1, 300, 2))
        cost, attraction, spread = action_labels(base, failed, candidate, failed, truth)
        np.testing.assert_allclose(cost, 0, atol=1e-14)
        np.testing.assert_allclose(attraction, spread)
        simultaneous = energy_costs(embedding(candidate[:, :, -1]), embedding(truth[:, -1]), failed[:, :, -1])[0]
        self.assertGreater(float(simultaneous[0]), 2.)
