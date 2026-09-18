import unittest
import numpy as np
from output_center_shift import attraction_gradient,shifted_cost
from audit_energy_action_value import embedding
from ensemble_score_objective import energy_costs


class OutputShiftTests(unittest.TestCase):
    def test_zero_exact_and_shift_equivalence(self):
        rng=np.random.default_rng(29);p=rng.normal(size=(6,8,2));y=rng.normal(size=(6,2));f=np.zeros((6,8),bool)
        np.testing.assert_array_equal(shifted_cost(p,y,f,[0.,0.]),energy_costs(embedding(p),embedding(y),f)[0])
        shift=np.array([.13,-.21])
        np.testing.assert_allclose(shifted_cost(p,y,f,shift),energy_costs(embedding(p+shift),embedding(y),f)[0],rtol=0,atol=1e-13)

    def test_gradient(self):
        rng=np.random.default_rng(33);p=rng.normal(size=(9,7,2));y=rng.normal(size=(9,2));s=np.array([.1,-.05]);eps=1e-6
        _,g=attraction_gradient(p,y,s)
        finite=np.stack([(attraction_gradient(p,y,s+eps*v)[0]-attraction_gradient(p,y,s-eps*v)[0])/(2*eps) for v in np.eye(2)],1)
        np.testing.assert_allclose(g,finite,atol=1e-9,rtol=1e-6)

    def test_exact_hit_finite_subgradient(self):
        a,g=attraction_gradient(np.zeros((1,3,2)),np.zeros((1,2)),[0.,0.])
        np.testing.assert_array_equal(a,[0.]);np.testing.assert_array_equal(g,np.zeros((1,2)))
