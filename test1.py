from steepest import *
import numpy as np


#          ROSENBROCK (TEST 1)

def funct(x:np.array):
    """Rosenbrock function"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0]**2) ** 2


def x1_1(x:np.array):
    return -400 * x[0] * (-x[0]**2 + x[1]) + 2*x[0] - 2


def x2_1(x:np.array):
    return -200 * x[0] ** 2 + 200 * x[1]



if __name__ == "__main__":
    '''Main'''

    # Test 1 (Rosenbrock)
    gradient = [x1_1, x2_1]
    print(steepest(funct, gradient, np.array([5, 3]), 0.1, 0.81, 0.0000001, 20, 100000))   # with the correct selection of parameters on this test, the method shows high accuracy
                                                                                           # nevertheless, this implementation requires quite a large number of iterations and it takes a few seconds



