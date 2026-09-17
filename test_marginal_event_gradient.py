import unittest
from audit_marginal_event_gradient import toy


class MarginalEventTests(unittest.TestCase):
    def test_conditional_score_and_full_expectation(self):
        for point in [(-.15,.07),(0.,0.),(.2,-.1)]:toy(point)


if __name__=='__main__':unittest.main()
