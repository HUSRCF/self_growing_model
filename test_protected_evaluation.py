import unittest
import numpy as np
from evaluate_protected_residual import distribution_scores
from evaluate_checked_particles import horizon_metrics


class ProtectedEvaluationTests(unittest.TestCase):
    def test_video_weighting_and_v_u_difference(self):
        rng=np.random.default_rng(11);p=rng.normal(size=(3,4,300,2))*.1
        truth=np.zeros((3,300,2));failed=np.zeros((3,4,300),bool);video=np.array([3,3,9])
        u=distribution_scores(p,truth,failed,video);v=horizon_metrics(p,truth,failed)
        for t in [50,100,300]:
            key=str(t);self.assertAlmostEqual(u[key]['mean'],(2*u[key]['per_video']['3']+u[key]['per_video']['9'])/3)
            x=p[:,:,t-1];emb=np.concatenate([np.sin(x),np.cos(x)],-1)
            total=np.linalg.norm(emb[:,:,None]-emb[:,None,:],axis=-1).sum((1,2)).mean()
            self.assertAlmostEqual(v[key]['energy_score']-u[key]['mean'],total/(2*4*4*3))
        zero=distribution_scores(np.zeros_like(p),truth,failed,video)
        self.assertTrue(all(z['mean']==0 for z in zero.values()))


if __name__=='__main__':unittest.main()
