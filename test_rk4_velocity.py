import unittest
import numpy as np
from rk4_velocity_pilot import rk4_step


class RK4Tests(unittest.TestCase):
    def test_constant_acceleration(self):
        y=np.array([[1.,2.]]);v=np.array([[.2,.3]]);a=np.array([[.01,-.02]])
        yp,vp=rk4_step(y,v,lambda pos,vel:a)
        np.testing.assert_allclose(yp,y+v+.5*a);np.testing.assert_allclose(vp,v+a)
        zero=rk4_step(y,v,lambda pos,vel:a,dt=0)
        np.testing.assert_array_equal(zero[0],y);np.testing.assert_array_equal(zero[1],v)

    def test_oscillator_order_and_no_mutation(self):
        errors=[]
        for n in [4,8]:
            y=np.array([[1.]]);v=np.array([[0.]])
            for _ in range(n):y,v=rk4_step(y,v,lambda pos,vel:-pos,dt=1/n)
            errors.append(np.linalg.norm([y.item()-np.cos(1),v.item()+np.sin(1)]))
        self.assertGreater(errors[0]/errors[1],14)
        y=np.ones((2,2));v=np.zeros((2,2));rk4_step(y,v,lambda pos,vel:-pos)
        np.testing.assert_array_equal(y,np.ones((2,2)));np.testing.assert_array_equal(v,np.zeros((2,2)))


if __name__=='__main__':unittest.main()
