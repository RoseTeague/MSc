#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import seaborn as sns

def Init(N):
    '''
    Initialised the positions of a chain of N argon atoms in a straight line

    Parameters
    -----------
    N - int, Number of argon atoms in the chain

    Returns
    --------
    pos - array, initial positions of argon atoms in chain
    v0 - array, initial velocity of each atom
    '''

    #Equilibrium separation
    do = 3.9 #Angstrom
    #Random number to determine an initial displacement from equilibrium
    sigma = np.sqrt(0.001)
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
    determines the force on each argon atom in the chain as a result of Lennard-
    Jones potentials

    Parameters
    -----------
    pos - array, initial positions of the argon atoms

    Returns
    -------
    a - array, acceleration of each atom
    V - Total potential energy (eV) for this arrangement
    F - array, the force on each atom as a result of interactions with every other
        atom in the chain

    '''

    A = 1.209*10**(5)#0.0011639175169510549#
    B = 70.66#0.7468261337975162#
    N = len(pos)

    #print(pos)

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
    Parameters
    ----------
    x - array
        positions of all atoms in chain
    '''
    N = len(x)
    r = [pos[n]-pos[n+1] for n in range(N-1)]

    return r


def velocityverlet(N, accel, x0, v0, dt, nt):
    '''Velocity Verlet integration to find x(t) and v(t)

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
        # Sometimes you'll see this with the two preceeding steps
        # combined to one, and a full step update of v
        # at the end
        a,V[it] = accel(x[:,it])
        # This is basically equivalent to the leapfrog method
        # except that we have v at that timestep at the end of
        # each step.
        v[:,it] = v[:,it] + 0.5*dt*a

        # plt.plot(x.transpose()[it], np.zeros(50), 'ro')
        # plt.show()
    return x,v,V

#sns.palplot(sns.color_palette("GnBu_d"))
def multiplot(N,x,y,l,cmap,plotname=None):
    plt.figure(N)
    sns.set_palette(cmap)
    for i in range(N):
        plt.plot(x,y[i]-y[i][0], linewidth = l)
    plt.savefig(plotname)


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

    #Unit conversions
    e = 1.6e-19
    A_to_m = 1e-10
    ps_to_s = 1e-12

    #Alternative units - to SI and back
    Fev_to_FSI = e/A_to_m #converting a force in eV per Angstrom to kg m s^-2
    aSI_to_aAps = ps_to_s**2/A_to_m # converting an acceleration from ms^-2 to Angstrom per ps^-2

    m = 6.63e-26/(Fev_to_FSI*aSI_to_aAps) # mass in eV ps^2 per Angstrom^2

    #Initialise time-step and duration, determine initial values and call differential equation solver
    nt = int(t/dt)
    tvals = np.linspace(0, t, nt)
    pos,v0 = Init(N)
    xvals,vvals,Vvals = velocityverlet(N,Accel,pos,v0,dt,nt)

    samples = np.zeros(nt)

    #plot displacement from initial position for each atom
    # for i in range(len(xvals)):
    #     plt.plot(tvals, xvals[i]-xvals[i][0], linewidth = 0.5)
    # plt.savefig('Displacement from initial position',dpi=700)
    multiplot(len(xvals),tvals,xvals,0.5,'GnBu_d','Displacement from initial position')

    #Calculate kinetic energy at each timestep as sum of individual atomic kinetic energies
    Tvals = np.array([0.5*m*np.sum(np.multiply(vvals[:,i],vvals[:,i])) for i in range(nt)])
    print('V = ', len(Vvals),' ',Vvals-Vvals[0], 'T = ', len(Tvals),' ',Tvals, 't = ', len(tvals))

    #Generate plots of potential, kinetic and total energies
    # plt.figure(6)
    # plt.plot(tvals, Vvals-Vvals[0],'r-.',linewidth = 0.2) #Sometimes this is a positive curve and sometimes it is negative.....
    # #plt.savefig('Potential Energy')
    # #plt.figure(7)
    # plt.plot(tvals, Tvals,'c-',linewidth = 0.2) #Tiimes by 10^-2 gives same order of magnitude as V but I don't know why..
    # #plt.savefig('Kinetic Energy')
    # #plt.figure(8)
    # plt.plot(tvals, Tvals+Vvals-Vvals[0], 'b--',linewidth = 0.2)
    # plt.savefig('Total Energy')

    Energies = [Vvals,Tvals,Vvals+Tvals]

    multiplot(3,tvals,Energies,0.2,'Set1','Energies')


    return Tvals+Vvals-Vvals[0]

if __name__=='__main__':
    #pos, v0 = Init(100)
    #a,V,r,F = Accel(pos)
    main(2,50,0.05)
