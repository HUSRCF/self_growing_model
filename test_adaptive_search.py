import unittest
import numpy as np
from adaptive_search_prototype import AdaptiveBeam, Node
from v20_rnn_mixture.engine.checker import central
from v20_rnn_mixture.engine.data import tail_windows


class PrototypeTests(unittest.TestCase):
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
