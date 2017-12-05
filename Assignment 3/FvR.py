#!/usr/bin/env python3

'''
Rosemary Teague - 00828351
Computational Methods 2 - Assignment 3
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.stats

def ParseFile(filename):
    '''
    Reads data from filename and stores values of the dynamical matrix as dynmat

    Parameters
    -----------
    filename - the name of the data file to be read

    Returns
    --------
    dynmat - a 6x6 dynamical matrix for the interaction between two atoms
    '''

    #Reads file and saves each line to file_lines
    f = open(str(filename), 'r')
    file_lines = f.readlines()
    f.close()

    #creates empty 6x6 dynamical matrix
    dynmat = np.empty((6, 6))

    #for each line containing data, the values of the dynamical matrix are saved
    #to  dynmat. File of the format that each row of the dynamical matrix is
    #given in one group with a blank line separating rows.
    for line in file_lines[5:]:
        val = line.split()
        if len(val) != 0:
            index1=int(val[0])+(int(val[1])-1)*3
            index2=int(val[2])+(int(val[3])-1)*3
            dynmat[index1-1,index2-1] = float(val[4])

    return dynmat


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

def Main():
    '''
    Caluculates the vibrational energy for 6 given dynamical matrices, each representing
    a bond-length between 2.0 and 2.5. Plots the correlation between the vibrational
    energies and the bond length with a linear fit.

    Parameters
    -----------
    NA

    Returns
    ---------
    slope - the slope of the fitted line
    intercept - the intercept of the fitted line
    r_value - the correlation coefficient of the fitted line
    std_err - Standard error of the estimated gradient.

    Saves a png image of the data points and a linearly fitted curve.

    Uses the scipy.stats.linregress function to produce linear fit.

    Atomic units are used througout

    '''

    #creates x data array of bond lengths in Bohn
    Bond_Lengths=np.linspace(2.0,2.5,6)
    #Creates list to contain Vibrational Energy values when calculated
    VibEnergy=[]
    #the names of the files containing the data to be analysed
    files=['ab_r2.0_dm','ab_r2.1_dm','ab_r2.2_dm','ab_r2.3_dm','ab_r2.4_dm','ab_r2.5_dm']

    #for each set of data, store the dynamical matrix, find the eigenvalues, and
    #determine the vibrational energy from these values.
    for file in files:
        dynmat=ParseFile(file)
        eigvals = scipy.linalg.eigvalsh(dynmat)
        VibFreq=np.sqrt(eigvals+0j)
        VibEnergy.append(0.5*np.real(VibFreq.max()))

    #Determine a linear fit and plot against the data points
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Bond_Lengths,VibEnergy)
    SavePlot(Bond_Lengths,intercept+slope*Bond_Lengths,Bond_Lengths,VibEnergy,xlabel='Bond Lengths (Bohr)', ylabel = 'Vibrational Energy (Hartree)',plotfilename='Vibrational Energy')

    #Output value of the slope to the terminal
    print('slope = ', slope)

    return [slope, intercept, r_value, std_err]

if __name__=='__main__':
    Main()
