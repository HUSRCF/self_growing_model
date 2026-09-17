import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io

from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import Actor, leave_one_out, future_costs, run_policy, prefix_windows, train_one


class ClosedLoopTests(unittest.TestCase):
    def test_accumulation_averages_before_update_and_preserves_budget(self):
        torch.set_num_threads(1)
        s=AdaptiveBeam();calls=[];validation=[]
        args=SimpleNamespace(epochs=10,accumulate=3,train_seed=1901,kl_weight=.01)
        def fake(s,windows,actor=None,training=False,seed=0,**kwargs):
            if not training:
                validation.append(seed)
                return dict(objective=1.,score={'100':{'embedding_rmse':1.}}),None,None
            calls.append((seed,actor.model[-1].bias.detach().clone()))
            loss=actor.model[-1].bias.sum()*((seed-42000+1)*.001)
            return loss,loss*0,dict(objective=1.,advantage_std=1.,failure=0.)
        with patch('train_closed_loop_policy.run_policy',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
            _,report=train_one(s,args,True)
        self.assertEqual([s for s,_ in calls],list(range(42000,42030)))
        self.assertEqual(len(validation),14)
        self.assertEqual(len(report['trace']),7)
        self.assertEqual(report['selected_epoch'],0)
        for i in range(10):
            for j in [1,2]:torch.testing.assert_close(calls[3*i][1],calls[3*i+j][1],rtol=0,atol=0)
            expected=np.sqrt(8)*(3*i+2)*.001
            self.assertAlmostEqual(report['train_trace'][i]['gradient_norm'],expected,places=6)
        self.assertFalse(torch.equal(calls[0][1],calls[3][1]))
        self.assertEqual(report['train_trace'][-1]['batches_seen'],30)

    def test_baseline_excludes_own_trajectory(self):
        a=np.array([1.,2.,3.,4.,8.,12.])
        np.testing.assert_allclose(leave_one_out(a,3),[2.5,2.,1.5,10.,8.,6.])
        b=a.copy();b[0]=1000
        self.assertEqual(leave_one_out(a,3)[0],leave_one_out(b,3)[0])

    def test_reward_to_go_excludes_past_horizons(self):
        r=future_costs([np.array([2.,4.]),np.array([6.,8.])],[2,4],4)
        np.testing.assert_array_equal(r,[[4,6],[4,6],[3,4],[3,4]])

    def test_zero_actor_and_truth_independence(self):
        torch.set_num_threads(1)
        s=AdaptiveBeam();w=prefix_windows([18],steps=50,per_video=2)
        actor=Actor(s,True)
        raw,p,f=run_policy(s,w,seed=1729,particles=2)
        _,p0,f0=run_policy(s,w,arrays=actor.arrays(),use_feedback=True,seed=1729,particles=2)
        np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(f,f0)
        altered={**w,'truth':w['truth']+1.}
        other,p1,f1=run_policy(s,altered,arrays=actor.arrays(),use_feedback=True,seed=1729,particles=2)
        np.testing.assert_array_equal(p0,p1);np.testing.assert_array_equal(f0,f1)
        self.assertNotEqual(raw['objective'],other['objective'])

    def test_policy_loss_has_gradient_without_changing_frozen_model(self):
        torch.set_num_threads(1)
        s=AdaptiveBeam();actor=Actor(s,True)
        frozen={k:v.copy() for k,v in s.machine.head.a.items()}
        w=prefix_windows([12],steps=50,per_video=2)
        loss,kl,stats=run_policy(s,w,actor=actor,training=True,particles=4)
        (loss+.01*kl).backward()
        self.assertTrue(np.isfinite(stats['objective']))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.model.parameters()),0.)
        for k,v in frozen.items():np.testing.assert_array_equal(v,s.machine.head.a[k])


if __name__=='__main__':unittest.main()
