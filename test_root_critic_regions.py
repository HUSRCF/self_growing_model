import unittest
import numpy as np
from confirm_root_critic_regions import summarize


class RootCriticRegionTests(unittest.TestCase):
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
