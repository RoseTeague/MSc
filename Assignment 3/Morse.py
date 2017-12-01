#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def Morse(r,De,a,re,E):
    '''
    Determines the Morse potential for an atomic separation r.

    Parameters
    ------------
    r - Atomic separation (Bohr)
    De - well depth (dissociation energy)
    a - width of potential
    re - the equilibrium bond distance
    b - scaling parameterim

    Returns
    --------
    V(r) - the Morse potential profile of atomic separation
    '''
    return De*(np.exp(-2*a*(r-re))-2*np.exp(-a*(r-re)))-E


def Main(filename):
    Data=np.loadtxt(str(filename))
    R=Data[:,0]
    E=Data[:,1]
    print(R[0],E[0])
    #
    # E=np.append(E,0)
    # R=np.append(R,100)

    Morse_guess = [E.min(),1,R[np.argmin(E)],E[-1]]
    Mp, cov = scipy.optimize.curve_fit(Morse, R, E, Morse_guess)

    print("De = ", Mp[0])
    print("a = ", Mp[1])
    print("re = ", Mp[2])
    print('E = ', Mp[3])

    xplt=np.linspace(R.min(),R.max(),100)

    plt.plot(R,E)
    plt.plot(xplt, Morse(xplt,*Mp))
    # plt.xlim(0.5,5)
    # plt.ylim(E.min(),E.max())
    plt.show()
    return str(filename)

if __name__=='__main__':
    Main(sys.argv[1])
