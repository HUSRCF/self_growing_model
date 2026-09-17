import itertools
import unittest
import numpy as np
from audit_particle_grouping import pooled_grouped_cost


class ParticleGroupingTests(unittest.TestCase):
    def test_all_balanced_partition_average_equals_pooled(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(2,6,4));y=rng.normal(size=(2,4))
        failed=rng.random((2,6))<.2;values=[]
        # All ten unordered partitions into two groups of three; particle0 anchors group1.
        for rest in itertools.combinations(range(1,6),2):
            group=np.array([0,*rest]);other=np.array([i for i in range(6) if i not in group])
            pooled,grouped=pooled_grouped_cost(x,y,failed,[group,other]);values.append(grouped)
        np.testing.assert_allclose(np.mean(values,axis=0),pooled,atol=1e-14)

    def test_invalid_partition_and_identical_particles(self):
        x=np.zeros((1,6,4));y=np.ones((1,4));failed=np.zeros((1,6),bool)
        a,b=pooled_grouped_cost(x,y,failed,[np.arange(3),np.arange(3,6)])
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):pooled_grouped_cost(x,y,failed,[np.arange(3),np.arange(3)])


if __name__=='__main__':unittest.main()
