import numpy as np
import matplotlib.pyplot as pyplot
import roots
import scipy

# Define the Lennard-Jones Potential for Argon
def Phi12(r):
    '''Lennard-Jones Potential for Argon (eV with r in Angstrom)'''
    return 1.209E5*r**(-12) - 70.66*r**(-6)

# Define the force associated with the Lennard-Jones potential for 2 Argon atoms
# separated by a distance r. This is calculcated from F=-grad(phi)
def Ar_Force(r):
    '''Lennard-Jones Force for Argon (eV with r in Angstrom)
        Determined as F=-grad(phi)
    '''
    return 1.209E5*12*r**(-13) - 70.66*6*r**(-7)

# find and output the equilibrium separation between a pair of Argon atoms
# such that the force is less than 10^-6 eV/Angstrom
def main():
    '''use the bisection method to find the equilibrium separation of two
    Argon atoms to within a tolerance of 10^-6 eV/Angstrom
    '''
    #Bisection Method
    r_equil=roots.bisection(func=Ar_Force, x0=3, x1=5, maxiter=100000, tol=10**(-6))
    #Scipy Brent's Method
    r_equil2=scipy.optimize.brentq(f=Ar_Force, a=3.0, b=5.0, xtol=1e-6)

    return 'Bisection Method:', r_equil, "Brent\'s Method:",r_equil2
