import sys
import numpy as np
import matplotlib.pyplot as plt

#!/usr/bin/env python3

def trapezoidal_np(dx, yvals):
    '''Calculate the area under y(x) using trapezoidal integration

    Parameters
    ----------
    dx - the spacing between x-values
    yvals - the value of y(x) at each x

    Returns
    --------
    area - numpy array of the cumulative area along x
    '''

    area = np.cumsum(yvals[:-1]+yvals[1:]) * 0.5 * dx
    return area


def main(filename, N):
    """ Plot a Density of States function and determine the energy of the highest
        occupied state for N electrons

    Parameters
    -----------
    filename - the name of the file containing Enegy and Density of States values
                for a sample
    N - the number of electrons

    Returns
    ----------
    Saves a plot of the Density of states as a function of energy
    """

    plt.close()
    #read the data file
    DOSE=np.loadtxt(str(filename))
    #extract energy and density of states values from the data file
    E=DOSE[:,0]
    DOS=DOSE[:,1]

    # calculate the area under the DOS plot using the trapezium rule and find the
    # value for which N electrons fill the lowest available energy state.
    dx = np.abs(E[0]-E[1])
    area=trapezoidal_np(dx,DOS)
    i=np.where(area>float(N))[0][0]
    Emax=E[i]

    # save a plot of the Density of states
    plt.plot(E,DOS)
    # Uncomment the following line to plot a vertical line at the energy of the highest occupied state
    #plt.axvline(x=Emax, ymin=0, ymax=1, linestyle='--')
    plt.xlabel('Energy')
    plt.ylabel('Density of States')
    plt.title('Electronic Density of States', fontsize=14)
    plt.text(min(E),1,'Energy of Highest Occupied Level \nfor '+str(N)+' electrons = '+str(Emax))
    plt.savefig('Density of States')


#Calls the main function with data input from unix terminal
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
