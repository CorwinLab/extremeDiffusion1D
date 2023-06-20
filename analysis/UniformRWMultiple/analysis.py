import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
import os

dir = '/home/jacob/Desktop/talapasMount/JacobData/UniformRWMultipleVs/'
mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

vs = list(mean.columns[2:])

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")

for v in vs: 
    if float(v) < 0.5:
        continue 
    ax.plot(mean['Time'], -mean[v], label=v, alpha=0.5)
    print(float(v) * mean['Time'].values **(3/4))
    print((float(v) * mean['Time'].values **(3/4)).astype(int))
    
ax.legend()
fig.savefig("MeanLogProb.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")

for v in vs:
    if float(v) < 0.5:
        continue
    ax.plot(var['Time'], var[v], label=v, alpha=0.5)

xvals = np.array([10**2, 10**4])
ax.plot(xvals, xvals**(-2/3)/100, ls='--', c='k', label=r'$t^{-2/3}$')

ax.legend()
fig.savefig("VarLogProb.png")

fig, ax = plt.subplots()
xvals = np.array([0.75, 1])

for v in vs: 
    if float(v) < 0.5:
        continue 
    ax.scatter(v, var[v].values[-1], c='k')
    print(v, var[v].values[-1])

fig.savefig("vDependence.png")