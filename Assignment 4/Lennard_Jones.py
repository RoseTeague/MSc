#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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
    sigma = np.sqrt(0.01)
    s = np.random.normal(0,sigma,N)

    #Generate positions chain of argon atoms
    pos = [do*n + s[n] for n in range(N)]
    pos = pos - pos[0]
    #initial velocity is zero
    v0 = np.zeros(N)

#    plt.plot(pos, np.zeros(N), 'ro')
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
    F - array, the force on each atom as a result of interactions with every other
        atom in the chain
    '''

    A = 1.209*10**(5)
    B = 70.66
    N = len(pos)

    #print(pos)

    #Matrix of interactions between each atomic pair
    Fmat = [[np.sign(n-m)*12*(A*((np.abs(pos[n]-pos[m]))**(-13))+B*0.5*((np.abs(pos[n]-pos[m]))**(-7))) for n in range(N) if n!=m]for m in range(N)]
    Vmat = [[(A*((np.abs(pos[n]-pos[m]))**(-12))+B*((np.abs(pos[n]-pos[m]))**(-6))) for n in range(N) if n!=m]for m in range(N)]

    F = [np.sum(Fmat[i]) for i in range(len(Fmat))]
    V = [np.sum(Vmat[i]) for i in range(len(Vmat))]
    #
    # plt.plot(pos, np.zeros(N), 'ro')
    # plt.plot(pos,F,'bx')
    # plt.show()

    m = 10**(-24)/(6.63*10**(-23))

    a = np.array(F)*m

    return a,np.array(V)*1.6*10


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
    x : float, array
        Position at each timestep.
    '''
    print(x0,dt,nt)
    # Initialize an empty arrays that will store x at each timestep.
    x = np.empty((N,nt))
    v = np.empty((N,nt))
    V = np.empty((N,nt))
    # Add in the initial conditions.
    x.transpose()[0] = x0
    v.transpose()[0] = v0
    # NB We only want to call this function once per cycle.
    a,V.transpose()[0] = accel(x.transpose()[0])

    # Use a for loop to iterate over timesteps, updating the velocity
    # and position each time.
    for it in range(1, nt):

        v.transpose()[it] = v.transpose()[it-1] + 0.5*dt*a
        x.transpose()[it] = x.transpose()[it-1] + v.transpose()[it]*dt
        # Sometimes you'll see this with the two preceeding steps
        # combined to one, and a full step update of v
        # at the end
        a,V.transpose()[it] = accel(x.transpose()[it])
        # This is basically equivalent to the leapfrog method
        # except that we have v at that timestep at the end of
        # each step.
        v.transpose()[it] = v.transpose()[it] + 0.5*dt*a
    return x,v,V

def Main(N,t,dt):
    '''
    Calulates the evolution of positions and velocities of a linear chain of argon
    atoms using the Lennard-Jones potential

    Parameters
    -----------
    N - Number of argon atoms (<100)
    t - duration of simulation (ps)
    dt - timestep used in integration of equation of motion (ps)
    '''
    m = (6.63*10**(-23))
    nt = int(t/dt)
    tvals = np.linspace(0, t, nt)
    pos,v0 = Init(N)
    xvals,vvals,Vvals = velocityverlet(N,Accel,pos,v0,dt,nt)


    # for i in range(len(xvals)):
    #     plt.plot(tvals, xvals[i]-xvals[i][0])
    # plt.show()

    for i in range(len(Vvals)-1):
        Tvals = 0.5*vvals[i]**2
        print(Tvals[4])
        plt.plot(tvals, Vvals[i],'r-.')
        plt.plot(tvals, Tvals,'c-')
        plt.plot(tvals, Tvals+Vvals[i], 'b--')
    plt.ylim(-1,5)
    plt.show()




if __name__=='__main__':
    Main(20,20,0.05)
