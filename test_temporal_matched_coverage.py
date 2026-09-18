import unittest
import numpy as np
from validate_temporal_matched_coverage import assert_purged,comparisons
from evaluate_conditional_policy import summarize


class TemporalCoverageTests(unittest.TestCase):
    def test_purging_history_and_target(self):
        train=dict(video=np.array([1,2]),start=np.array([100,200]))
        late=dict(video=np.array([1,2]),start=np.array([432,532]))
        assert_purged(train,late)
        with self.assertRaises(ValueError):assert_purged(train,dict(late,start=np.array([431,532])))
        with self.assertRaises(ValueError):assert_purged(train,dict(late,video=np.array([1,3])))

    def test_named_comparisons(self):
        names=['single_context','multi_context','single_action','multi_action','fixed_r6']
        mixed={n:np.full((2,3),i,dtype=float) for i,n in enumerate(names)}
        for m in mixed.values():m[:,0]=0
        row=dict(baseline=np.zeros((2,3)),mixed=mixed,
                 coefficients={n:dict(linear=m*4,quadratic=np.zeros_like(m)) for n,m in mixed.items()})
        rows=[row,row];v=np.array([1,2]);s=summarize(rows,v,('multi_context','single_context'))
        c=comparisons(rows,v)
        self.assertAlmostEqual(c['multi_context_minus_single_context']['delta'],2/3)
        self.assertAlmostEqual(s['context_minus_action'],2/3)
        self.assertAlmostEqual(c['multi_context_minus_fixed_r6']['delta'],-2.)


if __name__=='__main__':unittest.main()
