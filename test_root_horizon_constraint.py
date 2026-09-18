import unittest
import numpy as np
from audit_root_horizon_constraint import optimize_direction


class RootHorizonConstraintTests(unittest.TestCase):
    def test_zero_and_constraint_boundary(self):
        self.assertEqual(optimize_direction([1,-4,-4],[1,1,1])['alpha'],0.)
        result=optimize_direction([-.1,-1,-1],[1,1,1])
        self.assertAlmostEqual(result['alpha'],.1)
        self.assertLessEqual(result['short_violation'],1e-12)

    def test_degenerate_and_disconnected_feasibility(self):
        self.assertEqual(optimize_direction([0,-1,-1],[0,0,0])['alpha'],.25)
        # Concave short curve: zero and [.1,.25] feasible, positive near zero infeasible.
        result=optimize_direction([.1,1,1],[-1,-1,-1])
        self.assertEqual(result['alpha'],0.)
        self.assertEqual(optimize_direction([.1,-1,-1],[-1,-1,-1])['alpha'],.25)
        with self.assertRaises(ValueError):optimize_direction([np.nan,0,0],[0,0,0])

    def test_random_against_dense_feasible_grid(self):
        rng=np.random.default_rng(731)
        grid=np.linspace(0,.25,10001)
        for _ in range(50):
            l,q=rng.normal(size=(2,3))
            for constrained in [False,True]:
                result=optimize_direction(l,q,constrained)
                delta=grid[:,None]*l+grid[:,None]**2*q
                feasible=delta[:,0]<=1e-12 if constrained else np.ones(len(grid),bool)
                self.assertLessEqual(result['mean_delta'],delta[feasible].mean(1).min()+1e-12)
