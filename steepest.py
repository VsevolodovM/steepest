import numpy as np
import types
from numpy import linalg as LA
import random as RA


def gip(gradient, point): #gradient in point
    """This function returns value of gradient at the give point"""
    return np.array([f(point) for f in gradient])


def wolfe_powell(funct:types.FunctionType, gradient, x:np.array, d:np.array, t_init:float,  omega:float, rho:float):   # Algo 6.2 (Geiger)
    """This function returns parameter satisfying the conditions of Wolfe-Powell"""
    #Phase A
    t = t_init
    gamma = 1.1
    i = 0
    a, b = 0, 0

    while funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d or gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
        if funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d:
            a = 0
            b = t


            #Phase B
            tau1 = RA.uniform(0, 0.5)
            tau2 = RA.uniform(0, 0.5)
            a_j = a
            b_j = b
            j = 0
            t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))

            while funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d or gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
                if funct(x + t * d) >= funct(x) + omega * t * gip(gradient, x) @ d:
                    b_j = t
                    j += 1
                    t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))
                elif funct(x + t * d) < funct(x) + omega * t * gip(gradient, x) @ d and gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
                    a_j = t
                    j += 1
                    t = RA.uniform(a_j + tau1 * (b_j - a_j), b_j - tau2 * (b_j - a_j))
            return t

        elif funct(x + t * d) < funct(x) + omega * t * gip(gradient, x) @ d and gip(gradient, x + t * d) @ d < rho * gip(gradient, x) @ d:
            t = gamma * t
            i += 1

    return t


def steepest(funct:types.FunctionType, gradient:np.array, xinit:np.array, omega:float, rho:float, epsilon:float, M:float, maxit:int):  #Algo 8.1(Geiger)
    """This function is an implementation of Steepest Descent method"""
    case = 0
    k = 0
    x_k = xinit
    d_k = np.zeros_like(xinit)
    t_k = 0.001

    while LA.norm(gip(gradient, x_k)) > epsilon and k < maxit and LA.norm(x_k) < M:
        d_k = -gip(gradient, x_k) / LA.norm(gip(gradient, x_k))

        t_k = wolfe_powell(funct, gradient, x_k, d_k, t_k, omega, rho)

        x_k = x_k + t_k * d_k
        k += 1

    if LA.norm(gip(gradient, x_k)) <= epsilon:
        return [x_k, 0]
    elif k >= maxit:
        return [x_k, 1]
    elif LA.norm(x_k) >= M:
        return [x_k, 2]










