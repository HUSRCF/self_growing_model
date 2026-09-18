import unittest
from unittest.mock import patch
import numpy as np
from confirm_root_critic_regions import summarize,worker


class RootCriticRegionTests(unittest.TestCase):
    def test_particle_budget_passed_and_scores_unchanged(self):
        a=np.array([[1.,2.]]); b=np.zeros((1,2,2)); p=np.array([[.5,.5]])
        with patch('confirm_root_critic_regions.AdaptiveBeam'), patch('confirm_root_critic_regions.collect_moments') as collect:
            collect.return_value=(a,b,[(a,b)]*3,[0,0])
            args=('prefix',{},123,p,{'same':p})
            old=worker(args)
            self.assertEqual(collect.call_args.args[-1],16)
            new=worker(args+(32,))
            self.assertEqual(collect.call_args.args[-1],32)
            self.assertEqual(old,new)
            self.assertEqual(new['results']['same']['delta'],[0.])

    def test_paired_seed_and_video_summary(self):
        rows=[dict(baseline=[1.,2.],results={'a':dict(delta=d,horizon_delta=[sum(d)/2]*3)})
              for d in [[-1.,0.],[0.,3.]]]
        result=summarize(rows,np.array([18,13]))
        self.assertEqual(result['baseline'],1.5)
        s=result['strategies']['a']
        self.assertEqual(s['delta'],.5)
        self.assertEqual(s['seed_delta'],[-.5,1.5])
        self.assertAlmostEqual(s['conditional_seed_se'],1.)
        self.assertEqual(s['per_video_delta'],{'13':1.5,'18':-.5})
        self.assertEqual(s['better_seeds'],1)
