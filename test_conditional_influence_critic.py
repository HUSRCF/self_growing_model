import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from crossfit_conditional_influence import state_features,fit_folds
from validate_temporal_root_critic import score


class ConditionalCriticTests(unittest.TestCase):
    def test_fold_labels_and_serialization(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(6,8,4));y=rng.normal(size=(6,8));v=np.repeat([1,2,3],2)
        for context in [False,True]:
            folds,a=fit_folds(x,y,v,context);changed=y.copy();changed[:2]+=100*rng.normal(size=(2,8))
            _,b=fit_folds(x,changed,v,context)
            np.testing.assert_array_equal(a[:2],b[:2])
            replay=json.loads(json.dumps(folds))
            for vid,m in replay.items():np.testing.assert_array_equal(score(m,x[v==int(vid)]),a[v==int(vid)])

    def test_actual_memory_no_initialize_or_truth(self):
        def read(h,q,m):
            np.testing.assert_array_equal(m['hidden'],7)
            return np.ones((2,1)),np.full((2,1,8),1/8),{'read_hidden':m['hidden']+1}
        machine=SimpleNamespace(read=read,initialize=lambda h:self.fail('Must not initialize'))
        s=SimpleNamespace(machine=machine,base=SimpleNamespace(execute_rule=lambda h,q,r:h[:,-1]+r[:,None]*.001))
        state=dict(history=np.zeros((2,32,2)),q=np.array([1,2]),hidden=np.full((2,3),7),failed=np.zeros(2,bool))
        with patch('crossfit_conditional_influence.features',side_effect=lambda base,h,q,m:np.c_[h[:,-1],m]):
            x,prior=state_features(s,state)
        np.testing.assert_array_equal(x[:,:,2:5],8)
        np.testing.assert_array_equal(state['hidden'],7)
        np.testing.assert_array_equal(state['history'],0)
        np.testing.assert_array_equal(prior,1/8)
        self.assertEqual(x.shape,(2,8,13))


if __name__=='__main__':unittest.main()
