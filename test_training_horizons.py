import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import train_one,future_costs


class TrainingHorizonTests(unittest.TestCase):
    def test_eligibility_history_selection_and_budget_controls(self):
        records=[];reports=[]
        for steps,accumulate in [(100,1),(300,1),(100,3)]:
            calls=[]
            args=SimpleNamespace(epochs=1,accumulate=accumulate,train_seed=1901,kl_weight=.01,
                objective_kind='energy_u',selection_objective_kind='energy_u',window_sampling='uniform',
                train_steps=steps,eligible_steps=300,selection_steps=300,
                loss_horizons=[50,100] if steps==100 else [50,100,300],selection_horizons=[50,100,300])
            def fake(s,w,actor=None,training=False,**kwargs):
                calls.append((training,w['history'].copy(),w['start'].copy(),w['truth'].shape[1],kwargs['loss_horizons']))
                if not training:return dict(objective=1.,score={'100':{'embedding_rmse':1.}}),None,None
                loss=actor.model[-1].bias.sum()
                return loss,loss*0,dict(objective=1.,advantage_std=1.,failure=0.)
            with patch('train_closed_loop_policy.run_policy',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
                _,report=train_one(AdaptiveBeam(),args,False)
            self.assertTrue(all(c[3]==300 and c[4]==[50,100,300] for c in calls if not c[0]))
            self.assertTrue(all(c[3]==steps for c in calls if c[0]))
            records.append(next(c for c in calls if c[0]));reports.append(report)
        for a,b in zip(records,records[1:]):
            np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        self.assertEqual(reports[0]['simulated_training_steps']*3,reports[1]['simulated_training_steps'])
        self.assertEqual(reports[1]['simulated_training_steps'],reports[2]['simulated_training_steps'])

    def test_late_target_only_credits_remaining_actions(self):
        returns=future_costs([np.array([1.]),np.array([2.]),np.array([6.])],[50,100,300],300)
        self.assertEqual(returns[0,0],3.)
        self.assertEqual(returns[50,0],8/3)
        self.assertEqual(returns[100,0],2.)
        self.assertEqual(returns[299,0],2.)


if __name__=='__main__':unittest.main()
