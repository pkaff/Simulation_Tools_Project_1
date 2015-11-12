import unittest
import math
from BDF2_1 import *


class TestSpline(unittest.TestCase):
    #tests

    #test integrate function on BDF3 method for the linear test equation with lambda = -0.2
    def test_integrate3(self):
        def rhs(t,y):
            return -0.2*y
        y0 = 1
        t0 = 0
        tf = 1
        bdf = BDF_2(rhs, 3)
        t,y = integrate(t0, y0, tf, 0.001)
        res = 5*(1 - math.exp(-0.2))

        self.assertAlmostEqual(y(-1), res)

    #test integrate function on BDF4 method for the linear test equation with lambda = -0.2
    def test_integrate4(self):
        def rhs(t,y):
            return -0.2*y
        y0 = 1
        t0 = 0
        tf = 1
        bdf = BDF_3(rhs, 4)
        t,y = integrate(t0, y0, tf, 0.001)
        res = 5*(1 - math.exp(-0.2))

        self.assertAlmostEqual(y(-1), res)


if __name__ == '__main__':
    unittest.main()
