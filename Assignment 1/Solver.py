import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def Solver(Mdat,bdat):
    ''' Solves an equation of the form Mx=b

    Parameters
    -----------
    Mdat : Data file, must be entered in quotations, e.g Mdat='M.dat'
        contains n lines, each with n space-separated values,
        representing an nxn matrix to be solved.

    bdat : Data file, must be entered in quotations, e.g bdat='b.dat'
        contains n space-separated values representing an n-dimensional
        vector

    Returns
    --------
    x : an n-dimesional vector which is the solution to the equation Mx=b
    '''

    # Read in data from M.dat and store as a Matrix
    M=open(Mdat,'r')
    M_=[[int(value) for value in line.split()] for line in M]

    # Read in data from b.dat and store as a Vector
    b=open(bdat,'r')
    b=b.read()
    b_=[int(value) for value in b.split()]

    # Solve the equation Mx=b using the scipy.linalg function
    # Store solution as an array in x
    LU1, P1 = scipy.linalg.lu_factor(M_)
    x=scipy.linalg.lu_solve((LU1, P1), b_)

    # Checks solution (Should return zero.)
    # print(np.dot(M_,x)-b_)


    return x
