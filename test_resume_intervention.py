import copy
import unittest
from types import SimpleNamespace
import numpy as np
from audit_crossfit_value import continuation
from test_delayed_root import ToyMachine
from resume_intervention import resume, influence_cost


class ResumeTests(unittest.TestCase):
    def test_exact_suffix_and_immutable_state(self):
        s=SimpleNamespace(machine=ToyMachine(),base=SimpleNamespace(execute_rule=lambda h,q,r:h[:,-1]+r[:,None]*.001))
        snap={50:None};p,f=continuation(s,np.zeros((2,32,2)),13,particles=3,snapshots=snap)
        state=snap[50];old=copy.deepcopy(state)
        s.machine.initialize=lambda h: self.fail('Resume must not initialize memory')
        rp,rf=resume(s,state)
        np.testing.assert_array_equal(rp[:,0],p.reshape(6,300,2)[:,50:])
        np.testing.assert_array_equal(rf[:,0],f.reshape(6,300)[:,50:])
        for k in ['history','q','hidden','failed']:np.testing.assert_array_equal(state[k],old[k])
        self.assertEqual(state['rng_state'],old['rng_state'])
        a,_=resume(s,state,steps=4,root=6,seed=31,branches=4)
        b,_=resume(s,state,steps=4,root=6,seed=31,branches=4)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a[:,:,0]-state['history'][:,-1,None],.006,atol=1e-16)
        with self.assertRaises(ValueError):resume(s,state,branches=2)

    def test_influence_matches_energy_first_derivative(self):
        rng=np.random.default_rng(1);p=rng.normal(size=(1,7,4));q=rng.normal(size=(1,5,4));y=rng.normal(size=(1,4))
        # Exact finite-support probability distributions; all crosspairs included.
        def energy(alpha):
            x=np.concatenate([p[0],q[0]]);w=np.r_[np.full(7,(1-alpha)/7),np.full(5,alpha/5)]
            return w@np.linalg.norm(x-y[0],axis=1)-.5*w@np.linalg.norm(x[:,None]-x[None,:],axis=-1)@w
        derivative=influence_cost(q,y,p,np.zeros((1,5)))-influence_cost(p,y,p,np.zeros((1,7)))
        self.assertAlmostEqual(float(derivative[0]),(energy(1e-5)-energy(-1e-5))/2e-5,places=9)
        np.testing.assert_allclose(influence_cost(q,y,p,np.ones((1,5)))-influence_cost(q,y,p,np.zeros((1,5))),2.)


if __name__=='__main__':unittest.main()
