#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def ParseFile(filename):
    f = open(str(filename), 'r')
    file_lines = f.readlines()
    f.close()

    dynmat = np.empty((6, 6))
    row_counter = 0
    column_counter = 0
    for line in file_lines[5:]:
        val = line.split()
        if len(val) != 0:
            index1=int(val[0])+(int(val[1])-1)*3
            index2=int(val[2])+(int(val[3])-1)*3
            dynmat[index1-1,index2-1] = float(val[4])

    return dynmat


def SavePlot(x,y,x2=None,y2=None,xlabel=None,ylabel=None,plotfilename='plot.png'):
    plt.plot(x, y, "b-")
    if x2 is not None and y2 is not None:
        plt.plot(x2,y2,'rx')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(plotfilename)

def Main():

    Bond_Lengths=np.linspace(2.0,2.5,6)
    VibEnergy=[]
    files=['ab_r2.0_dm','ab_r2.1_dm','ab_r2.2_dm','ab_r2.3_dm','ab_r2.4_dm','ab_r2.5_dm']
    for file in files:
        dynmat=ParseFile(file)
        eigvals = scipy.linalg.eigvalsh(dynmat)
        VibEnergy.append(eigvals.max())


    SavePlot(Bond_Lengths,VibEnergy,xlabel='Bond Lengths', ylabel = 'Vibrational Energy',plotfilename='Vibrational Energy')

if __name__=='__main__':
    Main()
