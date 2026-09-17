import unittest
from dataclasses import replace
import numpy as np
from adaptive_search_prototype import AdaptiveBeam, Node, rollout
from v20_rnn_mixture.engine.checker import central
from v20_rnn_mixture.engine.data import tail_windows


class PrototypeTests(unittest.TestCase):
    def test_temperature_matching_preserves_root_prior_scale(self):
        h=np.zeros((1,32,2)); hidden=np.zeros((1,32))
        search=AdaptiveBeam(min_depth=2,max_depth=2,temperature=1,depth_invariant_temperature=True)
        roots=[Node(h,i,hidden,1,-float(2+2*i),i,i,i,i,path_q=(i,)) for i in range(2)]
        search._root_options=lambda *args:roots
        search._expand=lambda n:[replace(n,depth=n.depth+1,path_q=n.path_q+(n.q,))]
        class Capture:
            def choice(self,n,p):self.probs=p.copy();return 0
        rng=Capture()
        search.search(h,0,hidden,rng,allow_lookahead=True);deep=rng.probs.copy()
        search.search(h,0,hidden,rng,allow_lookahead=False)
        np.testing.assert_allclose(deep,rng.probs)
        search.depth_invariant_temperature=False
        search.search(h,0,hidden,rng,allow_lookahead=False)
        self.assertGreater(rng.probs[0],deep[0])

    def test_exact_rule_cache_preserves_rule_inputs(self):
        search=AdaptiveBeam(rule_cache_size=1)
        h=tail_windows('dev',10,1)['history'][:1]
        q=int(search.base.state_from_history(h)[0][0]);rs=np.array([0,1])
        first=search._execute_candidates(h,q,rs)
        np.testing.assert_array_equal(first,search._execute_candidates(h.copy(),q,rs))
        self.assertEqual(search.audit['rule_cache_hits'],1)
        search._execute_candidates(h,q,rs[::-1])
        self.assertEqual(search.audit['rule_cache_misses'],2)
        self.assertEqual(len(search._rule_cache),1)
        search._execute_candidates(h,q,rs)
        self.assertEqual(search.audit['rule_cache_misses'],3)

    def test_shallow_schedule_and_uncertainty_promotion(self):
        h=np.zeros((1,32,2)); hidden=np.zeros((1,32))
        for gap,expected in [(0.,1),(4.,2)]:
            search=AdaptiveBeam(min_depth=2,max_depth=2,uncertainty_gap=gap)
            roots=[Node(h,i,hidden,1,-float(1+3*i),i,i,i,i,path_q=(i,)) for i in range(2)]
            search._root_options=lambda *args:roots
            search._expand=lambda n:[replace(n,depth=n.depth+1,score=n.score-1,path_q=n.path_q+(n.q,))]
            _,info=search.search(h,0,hidden,np.random.default_rng(0),allow_lookahead=False)
            self.assertEqual(info['depth'],expected)

    def test_rollback_restores_parent_and_bans_dead_destination(self):
        class Fake:
            def __init__(self): self.machine=self
            def initialize(self,h): return np.array([0]),{'hidden':np.zeros((1,1))}
            def search(self,h,q,hidden,rng,check_root=True,banned_q=(),allow_lookahead=True):
                info=dict(depth=1,roots=1,stable=False,root_gap=None)
                if q==1:return None,info
                if q==0:
                    np.testing.assert_array_equal(hidden,np.zeros((1,1)))
                    r=2 if 1 in banned_q else 1
                else:r=2
                nh=np.concatenate([h[:,1:],np.full((1,1,2),r)],1)
                return Node(nh,r,hidden+1,1,0,0,r,0,r),info
        h=np.zeros((1,32,2))
        pred,failed,diag=rollout(Fake(),h,3,particles=1,rollback_budget=2)
        self.assertFalse(failed.any())
        np.testing.assert_array_equal(pred, np.full((1,1,3,2),2.))
        self.assertEqual(sum(d['rollback'] for d in diag),1)
        _,failed,_=rollout(Fake(),h,3,particles=1,rollback_budget=0)
        self.assertTrue(failed[0,0,1:].all())

    def test_support_stop_counts_unique_q_paths(self):
        h = np.zeros((1,32,2)); hidden = np.zeros((1,32))
        for required, expected in [(1,2),(2,3)]:
            search = AdaptiveBeam(min_depth=2,max_depth=3,min_stop_support=required)
            roots = [Node(h,i,hidden,1,-float(1+3*i),i,i,i,i,path_q=(i,)) for i in range(2)]
            search._root_options = lambda *args: roots
            # Duplicate event histories with the same q path must not count
            # as independent support for stopping.
            def expand(node):
                child=replace(node,depth=node.depth+1,score=node.score-(1+3*node.q),path_q=node.path_q+(node.q,))
                return [child,replace(child,event=7)]
            search._expand=expand
            _,info=search.search(h,0,hidden,np.random.default_rng(0))
            self.assertEqual(info['depth'],expected)

    def test_committed_middle_checked_after_replanning(self):
        search = AdaptiveBeam(min_depth=1, max_depth=1)
        h = tail_windows('dev', 10, 1)['history'][:1]
        q, memory = search.machine.initialize(h)
        search.checker.reject = lambda left, middle, right, qs: (np.ones(len(qs), bool), np.ones(len(qs)))
        node, _ = search.search(h, int(q[0]), memory['hidden'], np.random.default_rng(0), check_root=True)
        self.assertIsNone(node)
        observed, _ = search.search(h, int(q[0]), memory['hidden'], np.random.default_rng(0), check_root=False)
        self.assertIsNotNone(observed)

    def test_checker_receives_left_neighbor(self):
        search = AdaptiveBeam()
        h = np.stack([np.arange(32.) * .01] * 2, axis=-1)[None]
        q, memory = search.machine.initialize(h)
        node = Node(h, int(q[0]), memory['hidden'], 1, 0., 0, int(q[0]), 0, int(q[0]))
        calls = []
        def check(left, middle, right, qs):
            np.testing.assert_allclose(left[:, -1], np.repeat(h[:, -2], len(qs), axis=0))
            np.testing.assert_allclose(central(left, middle, right), (right - h[:, -2]) / .02)
            calls.append(len(qs))
            return np.zeros(len(qs), bool), np.zeros(len(qs))
        search.checker.reject = check
        search._expand(node)
        self.assertTrue(calls)

    def test_scalar_batch_equivalence(self):
        search = AdaptiveBeam()
        for h in tail_windows('dev', 10, 1)['history']:
            h = h[None]
            q, memory = search.machine.initialize(h)
            node = Node(h, int(q[0]), memory['hidden'], 1, 0., 0, int(q[0]), 0, int(q[0]))
            pe, trans, hidden = search._read(node)
            scalar = [search._child(node, int(e), int(r), pe, trans, hidden)
                      for e in search._top(pe, 2) for r in search._top(trans[e], 2)]
            scalar = [x for x in scalar if x is not None]
            batch = search._expand(node)
            self.assertEqual(len(scalar), len(batch))
            for a, b in zip(scalar, batch):
                self.assertEqual((a.event, a.q), (b.event, b.q))
                np.testing.assert_allclose(a.history, b.history, atol=1e-12)
                self.assertAlmostEqual(a.score, b.score, places=10)


if __name__ == '__main__':
    unittest.main()
