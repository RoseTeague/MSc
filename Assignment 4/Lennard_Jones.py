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
    F - array, the force on each atom as a result of interactions with every other
        atom in the chain
    '''

    A = 1.209*10**(5)#0.0011639175169510549#
    B = 70.66#0.7468261337975162#
    N = len(pos)

    #print(pos)

    #Matrix of interactions between each atomic pair
    rmat = np.array([[pos[n]-pos[m] for n in range(N) if n!=m] for m in range(N)])
    Vmat = A*np.abs(rmat[:,:])**(-12)-B*np.abs(rmat[:,:])**(-6)
    Fmat = -np.sign(rmat[:,:])*12*np.multiply(Vmat,1/np.abs(rmat))

    V = np.array([np.sum(Vmat[i]) for i in range(N)])
    F = np.array([np.sum(Fmat[i]) for i in range(N)])

    m_to_A = 10e10
    s_to_ps = 10e12
    kg_to_eVc = 1/(1.6e-19)#(3e8)**2/(1.6e-19)

    m = 6.63e-26*kg_to_eVc*10**(4)

    a = F/m

    plt.figure(1)
    plt.imshow(rmat)
    plt.colorbar()
    plt.show()

    plt.figure(2)
    plt.plot(pos,V)

    plt.figure(3)
    plt.plot(pos,F)

    plt.figure(5)
    plt.plot(pos,a)

    #Verification that force contributions are mainly from nearest neighbours
    plt.figure(4)
    plt.imshow(Fmat)
    plt.colorbar()
    plt.show()


    return a,2*np.sum(V)/N,rmat,Fmat

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
    #print(dt,nt)
    # Initialize an empty arrays that will store x at each timestep.
    x = np.zeros((N,nt))
    v = np.zeros((N,nt))
    V = np.zeros(nt)
    # Add in the initial conditions.
    x[:,0] = x0
    v[:,0] = v0
    # NB We only want to call this function once per cycle.
    a,V[0],r,F = accel(x0)

    # Use a for loop to iterate over timesteps, updating the velocity
    # and position each time.
    for it in range(1, nt):

        v[:,it] = v[:,it-1] + 0.5*dt*a
        x[:,it] = x[:,it-1] + v[:,it]*dt
        # Sometimes you'll see this with the two preceeding steps
        # combined to one, and a full step update of v
        # at the end
        a,V[it],r,F = accel(x[:,it])
        # This is basically equivalent to the leapfrog method
        # except that we have v at that timestep at the end of
        # each step.
        v[:,it] = v[:,it] + 0.5*dt*a

        # plt.plot(x.transpose()[it], np.zeros(50), 'ro')
        # plt.show()
    return x,v,V

def leapfrog(N,accel, x0, v0, dt, nt):
    '''Leapfrog integration to find x(t) and v(t)

    Parameters
    ----------
    N : integer
        Number of atoms in chain
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
    # Initialize an empty arrays that will store x at each timestep.
    x = np.empty((N,nt))
    # Add in the initial conditions.
    x = np.empty((N,nt))
    v = np.empty((N,nt))
    V = np.empty(nt)
    # Add in the initial conditions.
    x[:,0] = x0
    # Put velocity out of sync by half a step (leapfrog method
    a,V[0],r,F = accel(x0)
    v[:,0] = v0 + 0.5 * dt * a

    # Use a for loop to iterate over timesteps, updating the velocity
    # and position each time.
    for it in range(1, nt):
        x[:,it] = x[:,it-1] + dt * v[:,it-1]
        a,V[it],r,F = accel(x[:,it])
        v[:,it] = v[:,it-1] + dt * a

    # plt.plot(x[:,it], np.zeros(N), 'ro')
    # plt.show()
    return x,v,V

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

    nt = int(t/dt)
    tvals = np.linspace(0, t, nt)
    pos,v0 = Init(N)
    xvals,vvals,Vvals = velocityverlet(N,Accel,pos,v0,dt,nt)

    for i in range(len(xvals)):
        plt.plot(tvals, xvals[i]-xvals[i][0])
    plt.savefig('Displacement from initial position',dpi=700)

    Tvals = [0.5*m*np.sum((vvals[:,i])**2) for i in range(len(vvals[0]))]
    print('V = ', len(Vvals),' ',Vvals[3], 'T = ', len(Tvals),' ',Tvals[3], 't = ', len(tvals))
    print('V = ', Vvals-Vvals[0])
    print('T = ', Tvals)

    plt.close()
    plt.plot(tvals, Vvals-Vvals[0],'r-.')
    plt.savefig('Potential Energy')
    plt.plot(tvals, Tvals,'c-')
    plt.savefig('Kinetic Energy')
    plt.close()
    plt.plot(tvals, Tvals+Vvals, 'b--')
    plt.show()

if __name__=='__main__':
    pos, v0 = Init(100)
    #a,V,r,F = Accel(pos)
    x,v,V = velocityverlet(100, Accel, pos, v0, 0.05, 50)
    x2,v2,V2 = leapfrog(100, Accel, pos, v0, 0.05, 50)
