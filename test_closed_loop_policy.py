import unittest
import numpy as np
import torch

from adaptive_search_prototype import AdaptiveBeam
from train_closed_loop_policy import Actor, leave_one_out, future_costs, run_policy, prefix_windows


class ClosedLoopTests(unittest.TestCase):
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
