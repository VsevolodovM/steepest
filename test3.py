from steepest import *
import numpy as np


# EXPONENTIAL FUNCTION (TEST 3)

def funct(x:np.array):
     return -np.exp(- x[0] ** 2)

def grad(x:np.array):
     return 2 * x[0] * np.exp(-x[0] ** 2)





if __name__ == "__main__":
    '''Main'''
    gradient = [grad]  #in my implementation gradient has to be a type of list

    print(steepest(funct, gradient, np.array([1]), 0.1, 0.81, 0.000001, 20, 100))   #when choosing a point close enough to the minimum, the algorithm copes with finding the minimum of the function

    print(steepest(funct, gradient, np.array([3.9]), 0.1, 0.81, 0.000001, 20, 10000))

    print(steepest(funct, gradient, np.array([4]), 0.1, 0.81, 0.000001, 20, 10000000))  #starting from some point, the algorithm "gets stuck" at the starting point

