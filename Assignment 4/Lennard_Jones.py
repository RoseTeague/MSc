#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Init(N):
    '''
    ============================================================================
    Initialises the atomic positions of a chain of N argon atoms in 1D with a
    variance of 0.01 Angstrom - note if the variance is changed to 0.001
    Angstrom a much more periodic motion is observed as the atoms are initially
    more evenly spaced.
    ============================================================================

    Parameters
    -----------
    N : int,
        Number of argon atoms in the chain

    Returns
    --------
    pos : array,
        Initial positions of argon atoms in chain
    v0 : array,
        Initial velocity of each atom
    '''

    #Equilibrium separation
    do = 3.9 #Angstrom
    #Generates random number to determine an initial displacement from equilibrium with std dev sigma
    sigma = np.sqrt(0.01)
    s = np.random.normal(0,sigma,N)
    #s = np.zeros(N)

    #Generate positions chain of argon atoms
    pos = [do*n + s[n] for n in range(N)]
    pos = pos - pos[0]
    #initial velocity is zero
    v0 = np.zeros(N)

    # plt.plot(pos, np.zeros(N), 'ro')
    # plt.show()
    return np.array(pos),v0

def Accel(pos):
    '''
    ============================================================================
    determines the force on each argon atom in the chain as a result of Lennard-
    Jones potentials
    ============================================================================

    Parameters
    -----------
    pos : array
        Initial positions of the argon atoms

    Returns
    -------
    a : array
        Acceleration of each atom (Angstrom per ps^2)
    V : Float
        Total potential energy (eV) for this arrangement

    '''

    #Initialise Parameters
    A = 1.209*10**(5)
    B = 70.66
    N = len(pos)

    #Matrix of interactions between each atomic pair
    rmat = np.array([[pos[n]-pos[m] for n in range(N) if n!=m] for m in range(N)])
    #Lennard-Jones Potential generated between each atomic pair
    Vmat = A*np.abs(rmat[:,:])**(-12)-B*np.abs(rmat[:,:])**(-6) #
    #Force between each atomic pair
    Fmat = -np.sign(rmat[:,:])*12*(A*np.abs(rmat[:,:])**(-13)-0.5*B*np.abs(rmat[:,:])**(-7))

    #Total potential and force for each atom
    V = np.array([np.sum(Vmat[i]) for i in range(N)])
    F = np.array([np.sum(Fmat[i]) for i in range(N)])

    #Unit conversions
    e = 1.6e-19
    A_to_m = 1e-10
    ps_to_s = 1e-12

    #Alternative units - to SI and back
    Fev_to_FSI = e/A_to_m #converting a force in eV per Angstrom to kg m s^-2
    aSI_to_aAps = ps_to_s**2/A_to_m # converting an acceleration from ms^-2 to Angstrom per ps^-2

    m = 6.63e-26/(Fev_to_FSI*aSI_to_aAps) # mass in eV ps^2 per Angstrom^2
    a = F/m

    # Figures to illustrate the forces and potentials on each argon atoms

    # plt.figure(1)
    # plt.imshow(rmat)
    # plt.colorbar()
    # plt.show()

    # plt.figure(2)
    # plt.plot(pos,V)
    #
    # plt.figure(3)
    # plt.plot(pos,F)
    #
    # plt.figure(5)
    # plt.plot(pos,a)

    #Verification that force contributions are mainly from nearest neighbours
    # plt.figure(4)
    # plt.imshow(Fmat)
    # plt.colorbar()
    # plt.show()

    # diving the total potential by 2 eliminates double counting from pairs
    return a,np.sum(V)/2

def adjacent_pairs(x):
    '''
    ============================================================================
    Determines the separation between all sets of adjacent atoms in a chain
    ============================================================================

    Parameters
    ----------
    x : array
        positions of all atoms in chain

    Returns
    --------
    r : array
        separation of each set of adjacent pairs
    '''
    #Number of atoms in the chain
    N = len(x)
    #Calculate separation for only adjacent pairs
    r = [x[n+1]-x[n] for n in range(N-1)]

    return r


def velocityverlet(N, accel, x0, v0, dt, nt):
    '''
    ============================================================================
    Velocity Verlet integration to find x(t) and v(t)
    ============================================================================

    Parameters
    ----------
    accel : function
        Function of x that gives the acceleration.
    x0 : float
        Initial value of x.
    v0 : float
        Initial value of v.
    dt : float
        Timestep.
    nt : int
        Number of timesteps

    Returns
    -------
    x : float, (N x nt) array
        Position at each timestep.
    v : float, (N x nt) array
        Velocity at each timesteps
    V : float, (nt) array
        Potential energy at each timestep
    '''

    # Initialize an empty arrays that will store x,v and V at each timestep.
    x = np.zeros((N,nt))
    v = np.zeros((N,nt))
    V = np.zeros(nt)
    # Add in the initial conditions.
    x[:,0] = x0
    v[:,0] = v0
    # NB We only want to call this function once per cycle.
    a,V[0] = accel(x0)

    # Use a for loop to iterate over timesteps, updating the velocity
    # and position each time.
    for it in range(1, nt):

        v[:,it] = v[:,it-1] + 0.5*dt*a
        x[:,it] = x[:,it-1] + v[:,it]*dt

        a,V[it] = accel(x[:,it])

        v[:,it] = v[:,it] + 0.5*dt*a

    return x,v,V


def multiplot(N,x,y,l,cmap,plotname, xlabel, ylabel, relative_init=True, lim = None, legend = None):
    '''
    ============================================================================
    Plots multiple fucntions against the same y axis and saves figure as a png
    image
    ============================================================================

    Parameters
    -----------
    N : int
        Number of functions to be plotted
    x : array
        x data to be plotted
    y : nested arrays
        contains each set of y data to be plotted
    l : float
        linewidth of plot
    cmap : string
        colour map of plots
    plotname : string
        Name of final figure
    xlabel : string
    ylabel : string
    relative_init : boolean
        If set to true, a relative set of data will be plotted against the Initial
        value (y[i] - y[i][0] will be plotted)
    lim : Float
        upper x limit for plot data
    legend : array of strings
        contains the labels for each plot if legend is desired

    Returns
    --------
    saved png image of plot as 'plotname.png'
    '''

    if legend is None:
        legend = [None for i in range(N)]

    plt.figure(N)
    sns.set_palette(cmap)
    if relative_init:
        for i in range(N):
            plt.plot(x,y[i]-y[i][0], linewidth = l,label = legend[i])

    else:
        for i in range(N):
            plt.plot(x,y[i], linewidth = l,label = legend[i])


    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if lim is not None:
        plt.xlim(0,lim)
    if legend[0] is not None:
        plt.legend()
    if plotname is not None:
        plt.savefig(plotname)
    else:
        plt.savefig('Plot')

def Energy_plots(nt,Vvals,vvals,tvals):
    '''
    ============================================================================
    Detemines the Kinetic energy and generates a plot of Potential, Kinetic and
    Total Energies
    ============================================================================

    Parameters
    -----------
    nt : int
        number of data points
    Vvals : array
        potential energy data
    vvals : array
        velocity data
    tvals : array
        time data

    Returns
    --------
    Tvals : array
        Kinetic Energy data

    '''

    #Unit conversions
    e = 1.6e-19
    A_to_m = 1e-10
    ps_to_s = 1e-12

    #Alternative units - to SI and back
    Fev_to_FSI = e/A_to_m #converting a force in eV per Angstrom to kg m s^-2
    aSI_to_aAps = ps_to_s**2/A_to_m # converting an acceleration from ms^-2 to Angstrom per ps^-2

    m = 6.63e-26/(Fev_to_FSI*aSI_to_aAps) # mass in eV ps^2 per Angstrom^2

    #Calculate kinetic energy at each timestep as sum of individual atomic kinetic energies
    Tvals = np.array([0.5*m*np.sum(np.multiply(vvals[:,i],vvals[:,i])) for i in range(nt)])

    Energies = [Vvals,Tvals,Vvals+Tvals]

    #plot data
    multiplot(3,tvals,Energies,0.2,'Set1','Energies', xlabel = 'Time (ps)', ylabel = 'Energy (eV)',legend = ['Potential Energy','Kinetic Energy', 'Total Energy'])

    return Tvals

def Fourier(N,dt,nt,xvals):
    '''
    ============================================================================
    Calculates the fourier transform of the change in the separation between
    adjacent atoms relative to its average value
    ============================================================================

    Parameters
    ----------
    N : int
        Number of atoms in chain
    dt : float
        timestep intervals (ps) of data points
    nt : int
        total number of timesteps (ps)
    xvals : matrix array
        data of x positions of each atom at each timestep

    Returns
    --------
    ftr_avg : array
        list of real fourier transformed values
    ftr_avg : array
        list of imaginary fourier transformed values
    '''
    #Determine samples as values of the change in the separation between
    #adjacent atoms relative to its average value
    samples = np.zeros(nt)
    nn_separations = np.array([adjacent_pairs(xvals[:,i]) for i in range(nt)])
    nn_average = np.array([np.sum(nn_separations[:,i])/nt for i in range(N-1)])/nt
    samples = [nn_separations[:,i] - nn_average[i] for i in range(N-1)]

    #calculates the fourier transform of each pair
    ftr = np.zeros((N,int(nt/2+1)))
    fti = np.zeros((N,int(nt/2+1)))
    for i in range(N-1):
        ft_samples = np.fft.rfft(samples[i])
        ftr[i] = np.array(np.real(ft_samples))
        fti[i] = np.array(np.imag(ft_samples))

    #Average over all pairs
    ftr_avg = np.array([np.sum(ftr[:,i]) for i in range(len(ftr[0]))])/len(ftr[0])
    ftr_avg = ftr_avg.tolist()
    fti_avg = np.array([np.sum(fti[:,i]) for i in range(len(fti[0]))])/len(fti[0])
    fti_avg = fti_avg.tolist()

    #Determine set of frequencies corresponding to fourier transformed data
    freqs = np.fft.rfftfreq(len(samples[0]), dt)
    freqs = freqs.tolist()

    #remove the initial data point of each set
    ftr_avg.remove(ftr_avg[0])
    fti_avg.remove(fti_avg[0])
    freqs.remove(freqs[0])

    #plot data
    multiplot(2,freqs,[ftr_avg,fti_avg] ,0.5,'husl','Fourier Transform',relative_init=False,lim = 2, xlabel = 'Frequency (THz)', ylabel = None, legend=('Real','Imaginary'))

    return ftr_avg, fti_avg

def main(N,t,dt):
    '''
    Calulates the evolution of positions and velocities of a linear chain of argon
    atoms using the Lennard-Jones potential

    Parameters
    -----------
    N - Number of argon atoms (<100)
    t - duration of simulation (ps)
    dt - timestep used in integration of equation of motion (ps)

    '''

    #Initialise time-step and duration, determine initial values and call differential equation solver
    nt = int(t/dt)
    tvals = np.linspace(0, t, nt)
    pos,v0 = Init(N)
    xvals,vvals,Vvals = velocityverlet(N,Accel,pos,v0,dt,nt)

    #plot displacement from initial position of each atom
    multiplot(len(xvals),tvals,xvals,0.5,'GnBu_d','Displacement from initial position',xlabel = 'Time (ps)', ylabel = 'Displacement (Angstrom)')

    # calculate kinetic energy and plot all energies
    Tvals = Energy_plots(nt,Vvals,vvals,tvals)

    #perform a fourier transform and plot result
    ftr_avg, fti_avg = Fourier(N,dt,nt,xvals)



if __name__=='__main__':

    main(100,100,0.05)
