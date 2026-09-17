import itertools
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io
from ensemble_score_objective import energy_costs


class EnsembleObjectiveTests(unittest.TestCase):
    def test_leave_out_excludes_particle_and_all_incident_pairs(self):
        rng=np.random.default_rng(51);x=rng.normal(size=(2,4,4));y=rng.normal(size=(2,4));f=np.zeros((2,4))
        cost,base=energy_costs(x,y,f)
        for i in range(4):
            reduced=np.delete(x,i,axis=1);rf=np.delete(f,i,axis=1)
            expected,_=energy_costs(reduced,y,rf)
            np.testing.assert_allclose(base[:,i],expected,atol=1e-14)
            changed=x.copy();changed[:,i]+=7;ff=f.copy();ff[:,i]=1
            _,new_base=energy_costs(changed,y,ff)
            np.testing.assert_allclose(new_base[:,i],base[:,i],atol=1e-14)
        perm=[2,0,3,1];c,b=energy_costs(x[:,perm],y,f[:,perm])
        np.testing.assert_allclose(c,cost);np.testing.assert_allclose(b,base[:,perm])

    def test_exhaustive_score_function_matches_exact_gradient(self):
        # Bernoulli support{0,2}, truth0.7, P3. Expected energy is
        # .7 - 1.4*p + 2*p². Enumeration removes Monte Carlo uncertainty.
        theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True);p=theta.sigmoid()
        expected=0.;estimator=0.
        for bits in itertools.product([0,1],repeat=3):
            b=torch.tensor(bits,dtype=torch.float64)
            logp=b*torch.log(p)+(1-b)*torch.log1p(-p);prob=logp.sum().exp()
            x=2*np.array(bits,dtype=float).reshape(1,3,1)
            c,base=energy_costs(x,np.array([[.7]]),np.zeros((1,3)))
            advantage=torch.tensor(3*(c[:,None]-base)[0],dtype=torch.float64)
            expected=expected+prob*float(c[0])
            estimator=estimator+prob.detach()*(logp*advantage).mean()
        analytic=.7-1.4*p+2*p*p
        np.testing.assert_allclose(expected.detach(),analytic.detach(),atol=1e-14)
        g=torch.autograd.grad(expected,theta,retain_graph=True)[0]
        estimated=torch.autograd.grad(estimator,theta)[0]
        np.testing.assert_allclose(estimated,g,rtol=1e-12,atol=1e-12)

    def test_objective_does_not_change_frozen_inference(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import prefix_windows,run_policy
        s=AdaptiveBeam();w=prefix_windows([12],steps=100,per_video=1)
        a,p,f=run_policy(s,w,particles=4,objective_kind='mse')
        b,q,g=run_policy(s,w,particles=4,objective_kind='energy_u')
        np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(f,g)
        self.assertTrue(np.isfinite(b['objective']))
        self.assertEqual(a['score'],b['score'])
        _,q2,g2=run_policy(s,dict(w,truth=w['truth']+1),particles=4,objective_kind='energy_u')
        np.testing.assert_array_equal(q,q2);np.testing.assert_array_equal(g,g2)

    def test_energy_gradient_is_finite_and_leaves_backbone_frozen(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import Actor,prefix_windows,run_policy
        s=AdaptiveBeam();actor=Actor(s,True);before={k:v.copy() for k,v in s.machine.head.a.items()}
        w=prefix_windows([12],steps=50,per_video=2)
        loss,kl,stats=run_policy(s,w,actor=actor,training=True,particles=4,objective_kind='energy_u')
        (loss+.01*kl).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.model.parameters()),0.)
        for k,v in before.items():np.testing.assert_array_equal(v,s.machine.head.a[k])

    def test_invalid_shapes_and_small_ensemble(self):
        with self.assertRaises(ValueError):energy_costs(np.zeros((1,2,4)),np.zeros((1,4)),np.zeros((1,2)))
        with self.assertRaises(ValueError):energy_costs(np.zeros((1,4,4)),np.zeros((2,4)),np.zeros((1,4)))

    def test_selection_metric_does_not_change_training_objective(self):
        from adaptive_search_prototype import AdaptiveBeam
        from train_closed_loop_policy import train_one
        calls=[];args=SimpleNamespace(epochs=1,accumulate=1,train_seed=1901,kl_weight=.01,
                                     objective_kind='mse',selection_objective_kind='energy_u')
        def fake(s,windows,actor=None,training=False,objective_kind=None,**kwargs):
            calls.append((training,objective_kind))
            if not training:return dict(objective=1.,score={'100':{'embedding_rmse':1.}}),None,None
            loss=actor.model[-1].bias.sum()
            return loss,loss*0,dict(objective=1.,advantage_std=1.,failure=0.)
        with patch('train_closed_loop_policy.run_policy',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
            train_one(AdaptiveBeam(),args,False)
        self.assertEqual([kind for training,kind in calls if training],['mse'])
        self.assertTrue(all(kind=='energy_u' for training,kind in calls if not training))


if __name__=='__main__':unittest.main()
