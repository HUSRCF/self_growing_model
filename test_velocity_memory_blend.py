import unittest
import numpy as np
from velocity_memory_blend import BlendedWriter


class BlendTests(unittest.TestCase):
    def test_constant_acceleration_velocity_mapping_and_endpoint(self):
        class Base:
            def execute_rule(self,h,q,r):return h[:,-1]+.3
        class Weak:
            def initialize(self,h):return np.ones_like(h[:,-1])*.1
            def execute(self,h,q,r,v):return h[:,-1]+v+.02,v+.04
        h=np.zeros((2,32,2));v=np.ones((2,2))*.1
        for beta in [.25,.5,.75,1.]:
            model=BlendedWriter(Base(),Weak(),beta);y,nv=model.execute(h,None,None,v)
            np.testing.assert_allclose(y-h[:,-1],.5*(v+nv))
        y,nv=BlendedWriter(Base(),Weak(),1).execute(h,None,None,v)
        a,b=Weak().execute(h,None,None,v)
        np.testing.assert_array_equal(y,a);np.testing.assert_array_equal(nv,b)
        with self.assertRaises(ValueError):BlendedWriter(Base(),Weak(),0)


if __name__=='__main__':unittest.main()
