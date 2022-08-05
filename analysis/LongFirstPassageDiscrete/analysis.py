from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import glob
import os
import pandas as pd
from pyDiffusion.quadMath import prettifyQuad

directory = '/home/jacob/Desktop/talapasMount/JacobData/LongFirstPassageDiscrete7/Q*.txt'
files = glob.glob(directory)

def calculateMeanVar(files, maxDistance): 
    data_sum = None
    data_sum_squared = None
    number_of_files = 0
    for f in files: 
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        if data[-1, 0] !=maxDistance:
            continue

        if data_sum is None: 
            data_sum = data
        else:
            data_sum += data 
        
        if data_sum_squared is None: 
            data_sum_squared = data ** 2 
        else: 
            data_sum_squared += data ** 2
        number_of_files += 1

    print(number_of_files)
    mean = data_sum / number_of_files
    var = data_sum_squared / number_of_files - mean**2
    return mean, var

mean, var = calculateMeanVar(files, maxDistance=8059)
distance = mean[:, 0]
var = var[:, 1]
N = np.quad("1e7")
logN = np.log(N).astype(float)

alpha = 0.75

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.5, 500 ])
ax.set_xlabel(r"$x / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Min}})$")
ax.plot(distance / logN, var, alpha=alpha, label=prettifyQuad(N))

directory = '/home/jacob/Desktop/talapasMount/JacobData/LongFirstPassageDiscrete/Q*.txt'
files = glob.glob(directory)
mean, var = calculateMeanVar(files, maxDistance=27631)
distance = mean[:, 0]
var = var[:, 1]
N = np.quad("1e24")
logN = np.log(N).astype(float)
ax.plot(distance / logN, var, alpha=alpha, label=prettifyQuad(N))

ax.legend()
fig.savefig("Variance.pdf")