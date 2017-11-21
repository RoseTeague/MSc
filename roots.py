import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def bisection(func, x0, x1, maxiter, tol, full_output=False):
    '''
    Find the root of a function with the bisection method.

    Parameters
    ----------
    func : function
        A Python function we want to find a root of.
    x0, x1 : float
        The bounding values of the region to find the root.
    maxiter : integer
        The maximum number of iterations.
    tol : float
        Exit when the root is known within this tolerance

    Returns
    -------
    x : float
        Root of the function between x0 and x1.
    '''
    f0 = func(x0)
    f1 = func(x1)
    sign_f0 = f0 / abs(f0)
    sign_f1 = f1 / abs(f1)
    x_inter=np.empty(shape=[0,1])
    # Raise an error if invalid bounding values are given
    assert sign_f0 != sign_f1, "Error: func(x0) and func(x1) do not have opposite sign."

    # Repeat the convergence only for a maximum number of iterations, maxiter
    for i in range(maxiter):
        x = 0.5 * (x0 + x1)

        # form list containing sequence of convergence
        if full_output:
            x_inter=np.append(x_inter, x)

        fx = func(x)
        # Exit when we know f(x) within tol
        if abs(fx) < tol:
            # return x (or more information) when the function has converged
            print("bisection converged in", i, "iterations.")
            if full_output:
                return x, x_inter, fx
            else:
                return x

        sign_fx = fx / abs(fx)
        # Half the range as appropriate before the next iteration
        if sign_fx == sign_f0:
            x0 = x
        else:
            x1 = x

    # Raise an error if the function did not converge
    raise ValueError("Error: bisection failed to converge after max iterations.")
