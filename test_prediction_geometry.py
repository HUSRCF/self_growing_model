import unittest
import numpy as np
from audit_prediction_geometry import geometry,correlation


class PredictionGeometryTests(unittest.TestCase):
    def test_decomposition_and_periodicity(self):
        rng=np.random.default_rng(77);p=rng.normal(size=(5,7,2));y=rng.normal(size=(5,2));f=np.zeros((5,7),bool)
        a=geometry(p,y,f);b=geometry(p+2*np.pi,y-2*np.pi,f)
        np.testing.assert_allclose(a['particle_error'],a['center_error']+a['spread'],atol=1e-13)
        np.testing.assert_allclose(a['center_error_by_angle'].sum(1),a['center_error'],atol=1e-13)
        np.testing.assert_allclose(a['spread_by_angle'].sum(1),a['spread'],atol=1e-13)
        for k in ['score','center_error','spread','residual','resultant']:np.testing.assert_allclose(a[k],b[k],atol=1e-13)

    def test_identical_particles_and_degenerate_correlation(self):
        p=np.zeros((3,5,2));y=np.ones((3,2));d=geometry(p,y,np.zeros((3,5),bool))
        np.testing.assert_array_equal(d['spread'],np.zeros(3))
        np.testing.assert_array_equal(d['diversity'],np.zeros(3))
        self.assertIsNone(correlation(np.ones(4),np.arange(4)))

    def test_finite_particle_moment_correction_exact_exchangeability(self):
        # Enumerate all independent draws for truth plus five particles from a two-point law.
        bits=(np.arange(64)[:,None]>>np.arange(6))&1
        angles=(2*bits-1)*.2
        p=np.stack([angles[:,1:],np.zeros((64,5))],-1)
        y=np.stack([angles[:,0],np.zeros(64)],-1)
        d=geometry(p,y,np.zeros((64,5),bool))
        np.testing.assert_allclose(d['second_moment_excess'].mean(0),0,atol=1e-14)
