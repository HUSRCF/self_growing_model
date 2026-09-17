import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from adaptive_search_prototype import AdaptiveBeam,Node


class CommitProbeTests(unittest.TestCase):
    def setUp(self):
        self.h=np.zeros((1,32,2));self.hidden=np.zeros((1,32))

    def root(self,q,score,event=0):
        return Node(self.h,q,self.hidden,1,score,event,q,event,q,path_q=(q,),path_events=(event,))

    def test_dead_q_removes_duplicate_events_and_resamples(self):
        s=AdaptiveBeam(min_depth=2,max_depth=2,commit_probe=True)
        roots=[self.root(0,0),self.root(0,-.1,1),self.root(1,-1,2)]
        s._root_options=lambda *args:roots;calls=[]
        def expand(node,full=False):
            calls.append((node.q,full))
            return [] if node.q==0 else [replace(node,q=4,depth=2)]
        s._expand=expand
        node,info=s.search(self.h,0,self.hidden,np.random.default_rng(1),allow_lookahead=False)
        self.assertEqual(node.q,1);self.assertEqual(info['viability_witness_q'],4)
        self.assertEqual(calls,[(0,False),(0,True),(1,False)])
        self.assertEqual(s.audit['viability_rejected_roots'],2)
        np.testing.assert_array_equal(self.hidden,0)

    def test_probe_is_not_future_score_reranking_or_temperature_change(self):
        s=AdaptiveBeam(min_depth=2,max_depth=2,temperature=1,depth_invariant_temperature=True,commit_probe=True)
        s._root_options=lambda *args:[self.root(0,2),self.root(1,1)]
        s._expand=lambda node,full=False:[replace(node,depth=2,score=-1000 if node.q==0 else 1000)]
        class Capture:
            def choice(self,n,p):self.p=p.copy();return int(p.argmax())
        rng=Capture();node,info=s.search(self.h,0,self.hidden,rng,allow_lookahead=False)
        self.assertEqual(node.q,0)
        expected=np.exp(np.array([2.,1.])/2);expected/=expected.sum()
        np.testing.assert_allclose(rng.p,expected)
        self.assertEqual(info['scoring_depth'],1);self.assertEqual(info['depth'],2)

    def test_full_fallback_and_complete_exhaustion(self):
        for rescue in [False,True]:
            s=AdaptiveBeam(min_depth=2,max_depth=2,commit_probe=True)
            s._root_options=lambda *args:[self.root(0,1)]
            s._expand=lambda node,full=False:[replace(node,depth=2)] if full and rescue else []
            node,info=s.search(self.h,0,self.hidden,np.random.default_rng(1),allow_lookahead=False)
            self.assertEqual(node is not None,rescue)
            self.assertEqual(s.audit['viability_full_probes'],1)
            if not rescue:self.assertTrue(info['viability_exhausted'])

    def test_existing_deep_search_needs_no_additional_probe(self):
        s=AdaptiveBeam(min_depth=2,max_depth=2,commit_probe=True)
        s._root_options=lambda *args:[self.root(0,1)]
        s._expand=lambda node:[replace(node,depth=2)]
        node,info=s.search(self.h,0,self.hidden,np.random.default_rng(1),allow_lookahead=True)
        self.assertIsNotNone(node);self.assertEqual(s.audit['viability_probes'],0)
        self.assertTrue(info['viability_checked'])

    def test_fast_witness_matches_first_child_without_diagnosis(self):
        for soft in [False,True]:
            s=AdaptiveBeam(soft_check=soft)
            s._read=lambda node:(np.array([.6,.4]),np.array([[.9,.1],[.1,.9]]),self.hidden)
            s._execute_candidates=lambda *args:np.zeros((4,2))
            s.checker=SimpleNamespace(reject=lambda *args:(np.array([True,False,False,False]),np.array([np.inf,0.,0.,0.])))
            s.base.state_from_history=lambda h:(np.zeros(len(h),dtype=int),None)
            node=self.root(0,1)
            expected=s._expand(node)[0].q
            def forbidden(*args):raise AssertionError('unused q diagnosis executed')
            s.base.state_from_history=forbidden
            self.assertEqual(s._expand(node,viability_only=True),[expected])

    def test_fast_probe_empty_and_widening_preserve_witness_type(self):
        s=AdaptiveBeam(event_top=1,next_top=1,widen_on_empty=True)
        s._read=lambda node:(np.array([.6,.4]),np.array([[.9,.1],[.1,.9]]),self.hidden)
        s._execute_candidates=lambda h,q,rs:np.zeros((len(rs),2))
        s.checker=SimpleNamespace(reject=lambda left,middle,y,q:(np.arange(len(y))==0,np.zeros(len(y))))
        s.base.state_from_history=lambda *args: (_ for _ in ()).throw(AssertionError('diagnosis'))
        self.assertEqual(s._expand(self.root(0,1),viability_only=True),[1])
        self.assertEqual(s.audit['rescued'],1)
        s._execute_candidates=lambda h,q,rs:np.full((len(rs),2),np.nan)
        self.assertEqual(s._expand(self.root(0,1),viability_only=True),[])


if __name__=='__main__':unittest.main()
