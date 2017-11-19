import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Read in data from M.dat and store as a Matrix
M=open('M.dat','r')
M_=[[int(value) for value in line.split()] for line in M]

# Read in data from b.dat and store as a Vector
b=open('b.dat','r')
b=b.read()
b_=[int(value) for value in b.split()]

# Solve the equation Mx=b using the scipy.linalg function
# Store solution as an array in x
LU1, P1 = scipy.linalg.lu_factor(M_)
x=scipy.linalg.lu_solve((LU1, P1), b_)

print(x)
#Checks solution (Should return zero.)
print(np.dot(M_,x)-b_)
