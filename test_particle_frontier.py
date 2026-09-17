import unittest
import numpy as np
from evaluate_checked_particles import horizon_metrics,chunked_checks
from evaluate_adapted_checked import verify_checks
from v20_rnn_mixture.engine.evaluate import metrics
from v20_rnn_mixture.engine.checker import MixtureChecker
from summarize_particle_frontier import energy_u


class ParticleFrontierTests(unittest.TestCase):
    def test_off_diagonal_energy_formula(self):
        x=np.array([[0.,1.],[2.,3.],[-1.,.2]]);y=np.array([1.,1.])
        distance=np.linalg.norm(x-y,axis=-1).mean()
        pair=np.linalg.norm(x[:,None]-x[None],axis=-1)
        empirical=distance-.5*pair.mean()
        expected=distance-.5*pair.sum()/(len(x)*(len(x)-1))
        self.assertAlmostEqual(energy_u(empirical,distance,len(x)),expected)
        self.assertIsNone(energy_u(1.,1.,1))

    def test_horizon_scores_equal_full_scores_including_failures(self):
        rng=np.random.default_rng(72);pred=rng.normal(size=(3,7,300,2));truth=rng.normal(size=(3,300,2))
        failed=np.zeros((3,7,300),bool);failed[0,2,80:]=True
        full,_=metrics(pred,truth,failed);short=horizon_metrics(pred,truth,failed)
        for t in full:
            for key in full[t]:
                if full[t][key] is None:self.assertIsNone(short[t][key])
                else:np.testing.assert_allclose(short[t][key],full[t][key],rtol=1e-14,atol=1e-14)

    def test_chunked_check_matches_full_saved_valid_rollouts(self):
        from v20_rnn_mixture.engine.data import tail_windows
        a=np.load('adaptive_search_results/adapted_checked_model0_roll1729.npz');h=tail_windows('dev',300,8)['history']
        checker=MixtureChecker();args=[checker,h,a['prediction'],a['boundary'],a['states'],a['failed']]
        self.assertEqual(chunked_checks(*args,chunk=3),verify_checks(*args))


if __name__=='__main__':unittest.main()
