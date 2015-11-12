import unittest
import math
from BDF2_1 import *


class TestSpline(unittest.TestCase):
    #tests

    #test integrate function on linear test equation with lambda = -0.2
    def test_integrate(self):
        def rhs(t,y):
            return -0.2*y
        y0 = 1
        t0 = 0
        tf = 1
        bdf = BDF_2(rhs)
        t,y = integrate(t0, y0, tf, 0.001)
        res = 5*(1 - math.exp(-0.2))

        self.assertAlmostEqual(y(-1), res)

if __name__ == '__main__':
    unittest.main()
