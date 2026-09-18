import unittest
import numpy as np
from conditional_output_shift import objective,predict
from output_center_shift import attraction_gradient,shifted_cost
from audit_energy_action_value import embedding
from ensemble_score_objective import energy_costs


class ConditionalShiftTests(unittest.TestCase):
    def test_gradient(self):
        rng=np.random.default_rng(43);p=rng.normal(size=(11,6,2));y=rng.normal(size=(11,2))
        x=np.c_[np.ones(11),np.linspace(-2,2,11)];b=np.array([.02,-.03,.01,.01]);zero=attraction_gradient(p,y,[0.,0.])[0]
        _,g=objective(b,x,p,y,zero);eps=1e-6
        numeric=[(objective(b+eps*v,x,p,y,zero)[0]-objective(b-eps*v,x,p,y,zero)[0])/(2*eps) for v in np.eye(4)]
        np.testing.assert_allclose(g,numeric,atol=1e-9,rtol=1e-6)

    def test_window_shift_pair_invariance_and_bounds(self):
        rng=np.random.default_rng(49);p=rng.normal(size=(7,5,2));y=rng.normal(size=(7,2));f=np.zeros((7,5),bool)
        model=dict(mean=0.,scale=1.,beta=[[.1,-.1],[.25,.25]],use_shift=True)
        shift=predict(model,np.linspace(-10,10,7))
        self.assertTrue((np.abs(shift)<=.25).all())
        np.testing.assert_allclose(shifted_cost(p,y,f,shift[:,None]),energy_costs(embedding(p+shift[:,None]),embedding(y),f)[0],atol=1e-13,rtol=0)
        model['use_shift']=False;np.testing.assert_array_equal(predict(model,np.arange(7)),np.zeros((7,2)))
