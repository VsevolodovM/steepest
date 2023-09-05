from steepest import *
import numpy as np


#         QUADRATIC FUNCTION (TEST 2)

def funct(x:np.array):
    """Quadratic function of type 1/2 * x'*Q*x + c'* x"""
    Q = np.array([[14, 9, -1], [9, 18, 6], [-1, 6, 5]])
    c = np.array([1, 2, 3])
    return 0.5 * x @ Q @ x + c @ x

def row1(x:np.array):
    return 14 * x[0] + 9 * x[1] - x[2] + 1

def row2(x:np.array):
    return 9 * x[0] + 18 * x[1] + 6 * x[2] + 2

def row3(x:np.array):
    return -x[0] + 6 * x[1] + 5 * x[2] + 3


if __name__ == "__main__":
    '''Main'''


    gradient = [row1, row2, row3] #  Q*x + c

    print(steepest(funct, gradient, np.array([1, 2, 1]), 0.1, 0.81, 0.000001, 20, 1000))  #the algorithm copes with this task quite quickly
