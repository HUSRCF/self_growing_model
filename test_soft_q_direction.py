import unittest
import numpy as np
from audit_soft_q_direction import directions
from audit_soft_value import tilt


class SoftDirectionTests(unittest.TestCase):
    def test_conditional_vs_joint_and_zero_support(self):
        pe=np.array([[.3,.7]]);t=np.array([[[1.,0.,0.],[0.,1.,0.]]]);q=np.array([[0.,2.,7.]])
        ds,within,between=directions(pe,t,q)
        np.testing.assert_array_equal(ds['conditional'],0)
        np.testing.assert_allclose(ds['joint'],[[.42,-.42,0.]])
        np.testing.assert_allclose(within,0);np.testing.assert_allclose(between,.84)

    def test_finite_difference_and_constant_shift(self):
        pe=np.array([[.4,.6]]);t=np.array([[[.2,.8],[.7,.3]]]);q=np.array([[.1,.9]])
        ds,_,_=directions(pe,t,q);shifted,_,_=directions(pe,t,q+5)
        for k in ds:np.testing.assert_allclose(ds[k],shifted[k],atol=1e-14)
        h=1e-4;fd=np.einsum('ne,ner->nr',pe,(tilt(t,q,h)-tilt(t,-q,h))/(2*h))
        np.testing.assert_allclose(fd,ds['conditional'],atol=1e-9,rtol=0)
        with self.assertRaises(ValueError):directions(pe*2,t,q)


if __name__=='__main__':unittest.main()
