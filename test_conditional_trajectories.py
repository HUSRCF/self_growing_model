import unittest
import numpy as np
from audit_conditional_trajectories import energy_parts,group_summary
from ensemble_score_objective import energy_costs


class TrajectoryTests(unittest.TestCase):
    def test_parts_match_u_energy(self):
        rng=np.random.default_rng(3);p=rng.normal(size=(5,8,300,2));y=rng.normal(size=(5,300,2))
        f=rng.random((5,8,300))<.1;parts=energy_parts(p,y,f)
        for t in [0,49,99,299]:
            x=np.concatenate([np.sin(p[:,:,t]),np.cos(p[:,:,t])],-1)
            target=np.concatenate([np.sin(y[:,t]),np.cos(y[:,t])],-1)
            cost,_=energy_costs(x,target,f[:,:,t])
            np.testing.assert_allclose(parts['total'][:,t],cost,rtol=0,atol=1e-14)
        delta={k:np.stack([v,v*2]) for k,v in parts.items()}
        mask=np.arange(5)<2
        a=group_summary(delta,mask);b=group_summary(delta,~mask);whole=group_summary(delta,np.ones(5,bool))
        self.assertAlmostEqual(a['contribution']+b['contribution'],whole['mean_difference'])
        self.assertEqual(group_summary(delta,np.zeros(5,bool)),{'n':0})


if __name__=='__main__':unittest.main()
