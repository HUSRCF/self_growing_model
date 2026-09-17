import unittest
import numpy as np
from increment_residual_pilot import transport_targets


class IncrementTargetTests(unittest.TestCase):
    def test_identity_and_wrapping(self):
        h=np.zeros((2,3,2));h[:,-1]=[[3.13,-3.13],[.1,.2]]
        observed=dict(history=h,truth=np.array([[-3.13,3.13],[.2,.4]]),video=np.array([12,12]))
        identity=transport_targets(observed,observed)
        np.testing.assert_allclose(np.exp(1j*identity['truth']),np.exp(1j*observed['truth']))
        shifted={k:v.copy() for k,v in observed.items()};shifted['history']+=.7
        result=transport_targets(observed,shifted)
        np.testing.assert_allclose(result['truth']-identity['truth'],.7)
        self.assertLess(np.abs(result['truth'][0]-shifted['history'][0,-1]).max(),.03)
        np.testing.assert_array_equal(result['history'],shifted['history'])
        result['history']+=1
        np.testing.assert_allclose(shifted['history'],h+.7)

    def test_alignment_guard(self):
        data=dict(history=np.zeros((1,2,2)),truth=np.zeros((1,2)),video=np.array([12]))
        with self.assertRaises(ValueError):transport_targets(data,dict(data,video=np.array([18])))


if __name__=='__main__':unittest.main()
