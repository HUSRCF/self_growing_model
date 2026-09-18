import unittest
from types import SimpleNamespace
import numpy as np
from test_delayed_root import ToyMachine
from audit_crossfit_value import continuation
from resume_intervention import resume
from evaluate_conditional_policy import predict_choices,summarize


class ConditionalPolicyTests(unittest.TestCase):
    def test_vector_actions_and_scalar_parity(self):
        s=SimpleNamespace(machine=ToyMachine(),base=SimpleNamespace(execute_rule=lambda h,q,r:h[:,-1]+r[:,None]*.001))
        snap={50:None};continuation(s,np.zeros((2,32,2)),13,particles=1,snapshots=snap)
        state=snap[50]
        a,af=resume(s,state,root=6,steps=5);b,bf=resume(s,state,root=np.array([6,6]),steps=5)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(af,bf)
        p,_=resume(s,state,root=np.array([1,7]),steps=2,seed=15,branches=3)
        np.testing.assert_array_equal(p[0,:,0],np.full((3,2),.001));np.testing.assert_array_equal(p[1,:,0],np.full((3,2),.007))
        for bad in [np.array([1]),np.array([1.,2.]),np.array([0,8]),np.array([[1,2]])]:
            with self.assertRaises(ValueError):resume(s,state,root=bad)

    def test_video_fold_routing(self):
        model={'models':{'action':{'1':{'contextual':False,'scores':[0,1]},'2':{'contextual':False,'scores':[1,0]}}}}
        choices,_=predict_choices(model,np.zeros((3,2,4)),np.array([2,1,2]))
        np.testing.assert_array_equal(choices['action'],[1,0,1])

    def test_summary_linear_and_curvature(self):
        l=np.array([[0.,-.04,-.08],[0.,-.02,-.06]])
        q=np.array([[0.,.02,.03],[0.,.01,.02]])
        base=np.ones((2,3));mixed=base+.25*l+.0625*q
        names=['action','context','fixed_r6']
        row=dict(baseline=base.tolist(),mixed={k:mixed.tolist() for k in names},
                 coefficients={k:dict(linear=l.tolist(),quadratic=q.tolist()) for k in names})
        result=summarize([row,row],np.array([1,2]))
        for value in result['policies'].values():
            self.assertAlmostEqual(value['delta'],value['linear_contribution']+value['quadratic_contribution'])
            self.assertEqual(value['horizon_delta'][0],0.)
        self.assertEqual(result['context_minus_action'],0.)


if __name__=='__main__':unittest.main()
