import unittest
import numpy as np
from audit_residual_clipping import clipping_scores


class ClippingAuditTests(unittest.TestCase):
    def test_weighted_saturation_and_noop(self):
        centers=np.zeros((1,2,2));p=np.array([[.2,.8]])
        raw=np.array([[[.2,0],[0,0]]]);cap=np.array([.1,.1]);truth=np.zeros((1,2))
        z=clipping_scores(centers,truth,p,raw,cap)
        np.testing.assert_allclose(z['saturated_candidate_probability'],.2)
        np.testing.assert_allclose(z['saturated_coordinate_fraction'],.1)
        self.assertLess(z['bounded_embedding_mse'][0],z['raw_embedding_mse'][0])
        self.assertGreater(z['bounded_embedding_mse'][0],z['zero_embedding_mse'][0])
        z=clipping_scores(centers,truth,p,raw,np.ones(2))
        np.testing.assert_array_equal(z['bounded_embedding_mse'],z['raw_embedding_mse'])

    def test_clipping_can_remove_a_useful_correction(self):
        c=np.zeros((1,1,2));truth=np.full((1,2),.2);p=np.ones((1,1))
        z=clipping_scores(c,truth,p,np.full_like(c,.2),np.full(2,.1))
        self.assertEqual(z['raw_embedding_mse'][0],0)
        self.assertGreater(z['bounded_embedding_mse'][0],0)
        self.assertGreater(z['raw_first_order_gain'][0],0)
        tiny=clipping_scores(c,truth,p,np.full_like(c,2e-7),np.ones(2))
        np.testing.assert_allclose(tiny['zero_embedding_mse']-tiny['raw_embedding_mse'],
                                   tiny['raw_first_order_gain'],rtol=1e-5)


if __name__=='__main__':unittest.main()
