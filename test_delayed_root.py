import unittest
from types import SimpleNamespace
import numpy as np
from audit_crossfit_value import continuation


class ToyMachine:
    def initialize(self,h):return np.zeros(len(h),int),{'hidden':np.zeros((len(h),1))}
    def read(self,h,q,m):
        tr=np.zeros((len(h),1,2));tr[:,:,0]=1
        return np.ones((len(h),1)),tr,{'read_hidden':m['hidden']+1}


class DelayedRootTests(unittest.TestCase):
    def test_first50_noop_and_default(self):
        engine=SimpleNamespace(machine=ToyMachine(),base=SimpleNamespace(execute_rule=lambda h,q,r:h[:,-1]+.001*r[:,None]))
        h=np.zeros((2,32,2));base,bf=continuation(engine,h,17,particles=3)
        p,f=continuation(engine,h,17,particles=3,root=1,forced_step=50)
        np.testing.assert_array_equal(p[:,:,:50],base[:,:,:50]);np.testing.assert_array_equal(f,bf)
        np.testing.assert_allclose(p[:,:,50],.001)
        no_op,nf=continuation(engine,h,17,particles=3,forced_step=50)
        np.testing.assert_array_equal(no_op,base);np.testing.assert_array_equal(nf,bf)
        early,_=continuation(engine,h,17,particles=3,root=1)
        np.testing.assert_allclose(early[:,:,0],.001)
        for t in [-1,300,.5]:
            with self.assertRaises(ValueError):continuation(engine,h,17,forced_step=t)
