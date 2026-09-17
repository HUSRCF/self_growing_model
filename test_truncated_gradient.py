import unittest
from audit_truncated_gradient import toy


class TruncatedGradientTests(unittest.TestCase):
    def test_bias_is_not_hidden(self):
        for p in [(-.15,.07),(0.,0.),(.2,-.1)]:self.assertGreater(toy(p)['bias_norm'],1e-8)


if __name__=='__main__':unittest.main()
