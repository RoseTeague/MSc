#!/usr/bin/env python3

'''
Rosemary Teague - 00828351
Computational Methods 2 - Assignment 3
'''

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
    E - scaling parameter to account for non-zero behaviour at large R

    Returns
    --------
    V(r) - the Morse potential profile of atomic separation
    '''

    V = De*(np.exp(-2*a*(r-re))-2*np.exp(-a*(r-re)))-E
    return V

def SavePlot(x,y,x2=None,y2=None,xlabel=None,ylabel=None,plotfilename='plot.png'):
    '''
    Creates png image of a plot containing one or two data sets

    Parameters
    -------------
    (Required)
    x - 1st set of x data (fitted curve)
    y - 1st set of y data (fitted curve)

    (Optional)
    x2 - 2nd set of x data (experimental points)
    y2 - 2nd set of y data (experimental points)
    xlabel - label for horizontal axis
    ylabel - label for vertical axis
    plotfilename - name of the image to be saved

    '''
    plt.plot(x, y, "b-")

    #Plot second set of data points if available
    if x2 is not None and y2 is not None:
        plt.plot(x2,y2,'rx')
    #Add x-axis label if given
    if xlabel is not None:
        plt.xlabel(xlabel)
    #Add y-axis label if given
    if ylabel is not None:
        plt.ylabel(ylabel)
    #Saves plot as png
    plt.savefig(plotfilename)


def Main(filename):
    '''
    Reads data for the energy and separation of a diatomic molecule and fits the
    Morse potential to this data.

    Parameters
    -----------
    filename - the name of the file containing Energy and Separation data to be readlines

    Returns
    --------
    Mp - a list containing the fitted parameters of the potential

    Each of the fitted parameters will be printed to the terminal and a png image
    of the data and interpolated plot will be saved as 'Morse Potential.png'
    '''

    #Reads the data from given file and stores Energy and Separation values as seperate lists
    Data=np.loadtxt(str(filename))
    R=Data[:,0]
    E=Data[:,1]

    #Gives initial guesses for the fitting parameters from the data points
    Morse_guess = [E.min(),1,R[np.argmin(E)],E[-1]]
    #Fits a curve of the form of the Morse Potential to the data points
    Mp, cov = scipy.optimize.curve_fit(Morse, R, E, Morse_guess)

    #Outputs the final fitted parameters to the terminal
    print("De = ", Mp[0])
    print("a = ", Mp[1])
    print("re = ", Mp[2])
    print('E = ', Mp[3])

    #saves a plot of the data points and fitted curve as a png image
    xplt=np.linspace(R.min(),R.max(),100)
    SavePlot(xplt,Morse(xplt,*Mp),R,E,xlabel='Atomic Separation',ylabel='Potential',plotfilename='Morse Potential')

    return Mp

if __name__=='__main__':
    Main(sys.argv[1])
