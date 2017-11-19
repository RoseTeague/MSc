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
    # Lets be sure we have valid x0 and x1 before proceeding.
    # We can use assert to raise an error if a condition isn't met.
    assert sign_f0 != sign_f1, "Error: func(x0) and func(x1) do not have opposite sign."

    # It's usually a good idea to enforce a limit on how many iterations a loop can
    # do. Otherwise if something goes wrong, it may take some time to understand that
    # your code is not converging.
    for i in range(maxiter):
        x = 0.5 * (x0 + x1)

        if full_output:
            x_inter=np.append(x_inter, x)

        fx = func(x)
        # Exit when we know f(x) within tol
        if abs(fx) < tol:
            # You can print out some info if you like. This is just for information
            # in this example. You may not want output from functions cluttering a
            # bigger code, and this will interfere with the use of %timeit
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

    # If we get to here without returning a value, we haven't converged.
    # "raise" is another good way to get your code to exit with an informative error if
    # an input is not correct. There's a list of exceptions that can be raised at
    # https://docs.python.org/3.6/library/exceptions.html
    raise ValueError("Error: bisection failed to converge after max iterations.")
