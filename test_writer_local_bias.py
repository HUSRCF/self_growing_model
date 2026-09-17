import unittest
import numpy as np
from audit_writer_local_bias import diagnostic_data,decompose


class LocalBiasTests(unittest.TestCase):
    def test_prefix_and_quadratic_proxy(self):
        t=np.arange(200,dtype=float);y=np.c_[.001*t*t,.02*t]
        h,truth,v,a=diagnostic_data(y)
        np.testing.assert_allclose(h[:,-1]+v+.5*a,truth,atol=1e-11)
        altered=y.copy();altered[100:]+=100
        for old,new in zip(diagnostic_data(y),diagnostic_data(altered)):np.testing.assert_array_equal(old,new)

    def test_residual_decomposition(self):
        r=np.random.default_rng(4);v,pv,acc,pa,current,truth=r.normal(size=(6,10,2))
        terms=decompose(v,pv,acc,pa,current,truth)
        np.testing.assert_allclose(terms.sum(1),current+v+.5*acc-truth,atol=1e-14)


if __name__=='__main__':unittest.main()
