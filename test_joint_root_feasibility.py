import unittest
import numpy as np
from audit_joint_root_feasibility import minimax,finite_bound


class JointRootFeasibilityTests(unittest.TestCase):
    def test_complementary_directions(self):
        # Neither single root is shared descent; their mixture is.
        result=minimax(np.array([[-2.,1.],[1.,-2.]]))
        np.testing.assert_allclose(result['direction'],[.5,.5],atol=1e-12)
        self.assertAlmostEqual(result['upper'],-.5)
        self.assertAlmostEqual(result['lower'],-.5)

    def test_positive_local_and_finite_curvature(self):
        l=np.eye(2);result=minimax(l)
        self.assertAlmostEqual(result['lower'],.5)
        self.assertTrue(finite_bound(l,np.zeros((2,2,2)),result['video_weights'])['excludes_nonzero_to_limit'])
        q=np.broadcast_to(-3*np.eye(2),(2,2,2))
        self.assertFalse(finite_bound(l,q,result['video_weights'])['excludes_nonzero_to_limit'])
        # A failed sufficient bound is NOT a feasibility declaration.

    def test_finite_lower_bound_on_random_simplex(self):
        rng=np.random.default_rng(762)
        l=rng.normal(size=(4,3));q=rng.normal(size=(4,3,3));q=(q+q.transpose(0,2,1))/2
        weights=rng.dirichlet(np.ones(4));bound=finite_bound(l,q,weights)
        for _ in range(100):
            s=rng.dirichlet(np.ones(3));alpha=rng.uniform(0,.25)
            actual=weights@(alpha*(l@s)+alpha**2*np.einsum('r,vrs,s->v',s,q,s))
            self.assertGreaterEqual(actual,alpha*bound['finite_margin']-1e-12)
