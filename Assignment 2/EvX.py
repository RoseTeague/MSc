#!/usr/bin/env python3
'''Rosemary Teague, CID - 00828351
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import sys

def main(filename):
    '''
    Finds the minimum energy separation of a diatomic molecule using input Data
        file

    Parameters
    ----------
    filename : input data file
        This should contain calculated values of the energy vs. separation of a
        diatomic molecule, with the energies in Rydberg and separations in Bohr:
        each line of the file should list a separation followed by an energy
        with a space in between.

    Saves a plot of the energy vs separation as a png file
    '''
    plt.close()

    #read the data file
    Etot=np.loadtxt(str(filename))
    #extract energy and separation values from the file data
    E=Etot[:,1]
    R=Etot[:,0]

    #Interpolate the data points
    interp=scipy.interpolate.InterpolatedUnivariateSpline(R,E)
    Xvals=np.linspace(R[0],R[-1],10*len(R))
    #Finds the minimum energy of interpolated Data and the corresponding separation
    Emin=min(interp(Xvals))
    Rmin=Xvals[np.argmin(interp(Xvals))]

    #Checks if minimum is at either end of the input data
    dr=R[0]-R[1]
    if Rmin<=(R[0]+dr) or Rmin>=(R[-1]-dr):
        # returns a warning that the spline may not have interpolated correctly
        return 'Warning: minimum at beginning/end of data, possible fault in interpolation'
    else:
        #saves plot of energy vs separation
        plt.plot(Xvals,interp(Xvals)-Emin)
        plt.xlabel('Separation (Bohr)')
        plt.ylabel('Energy Relative to minimum energy (Rydberg)')
        plt.title('Separation Energy of a diatomic Molecule',fontsize=14)
        plt.text(R[0],max(interp(Xvals))-Emin,'Minimum Energy Separation = '+str(Rmin)+'(Bohr)',verticalalignment='top')
        plt.savefig('Separation Energy')




#Calls the main function with data input from unix terminal
if __name__ == '__main__':
    main(sys.argv[1])
