"""Exact frozen event-read cache isolation, lifetime and immutability."""
import unittest
from dataclasses import replace
import numpy as np
from adaptive_search_prototype import AdaptiveBeam,Node


class FakeReader:
    def __init__(self):self.calls=0;self.bias=0.
    def read(self,h,q,memory):
        self.calls+=1
        value=float(h.sum()+q.sum()+memory['hidden'].sum()+self.bias)
        self.last=(np.full((1,2),value),np.full((1,2,2),value),np.full((1,3),value))
        return self.last[0],self.last[1],{'read_hidden':self.last[2]}


class ReadCacheTests(unittest.TestCase):
    def setUp(self):
        self.s=AdaptiveBeam(read_cache_size=2);self.s.machine=FakeReader()
        self.n=Node(np.zeros((1,32,2)),0,np.zeros((1,3)),0,0.,-1,-1,-1,-1)

    def test_exact_repeat_and_readonly_owned_results(self):
        a=self.s._read(self.n);b=self.s._read(replace(self.n,history=self.n.history.copy(),hidden=self.n.hidden.copy()))
        self.assertEqual(self.s.machine.calls,1)
        self.assertEqual(self.s.audit['read_cache_hits'],1)
        for x,y in zip(a,b):
            self.assertIs(x,y)
            with self.assertRaises(ValueError):x.flat[0]=1
        self.s.machine.last[0][:]=9
        np.testing.assert_array_equal(a[0],0)
        np.testing.assert_array_equal(self.n.hidden,0)

    def test_key_includes_history_q_hidden_shape_and_dtype(self):
        variants=[self.n,replace(self.n,q=1),replace(self.n,hidden=np.ones((1,3))),
                  replace(self.n,history=np.ones((1,32,2))),
                  replace(self.n,hidden=np.zeros((1,3),dtype=np.float32)),
                  replace(self.n,hidden=np.zeros((3,1)))]
        for n in variants:self.s._read(n)
        self.assertEqual(self.s.machine.calls,len(variants))

    def test_lru_eviction_and_explicit_model_update_invalidation(self):
        self.s._read(self.n);self.s._read(replace(self.n,q=1));self.s._read(self.n)
        self.s._read(replace(self.n,q=2));self.s._read(replace(self.n,q=1))
        self.assertEqual(self.s.machine.calls,4)
        self.assertEqual(len(self.s._read_cache),2)
        self.s.machine.bias=7;self.s.clear_read_cache()
        np.testing.assert_array_equal(self.s._read(self.n)[0],7)
        reader=FakeReader();reader.bias=9;self.s.machine=reader
        np.testing.assert_array_equal(self.s._read(self.n)[0],9)

    def test_disabled_cache_reads_every_time(self):
        self.s.read_cache_size=0
        self.s._read(self.n);self.s._read(self.n)
        self.assertEqual(self.s.machine.calls,2)
        self.assertFalse(self.s._read_cache)
        with self.assertRaises(ValueError):AdaptiveBeam(read_cache_size=-1)


if __name__=='__main__':unittest.main()
