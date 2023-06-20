import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import json

dir = '/home/jacob/Desktop/talapasMount/JacobData/UniformRWLonger/'

mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

step_size = 10

def theoreticalMean(v, time, step_size):
    sigma2 = 1/9 * step_size * (step_size + 2)
    return v**2 * np.sqrt(time) / 2 / sigma2 

vs = list(mean.columns[2:])
final_x = np.array(vs).astype(float) * max(mean['Time'])**(3/4)
exponent = final_x ** 2 / 2 / max(mean['Time'])
vs_accepted = []
for i in range(len(vs)):
    if exponent[i] > 10:
        vs_accepted.append(vs[i]) 
    
vs = vs_accepted

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(vs) / 1) for i in range(len(vs))]

''' Make the Mean Plot '''
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([min(mean['Time']), max(mean['Time'])])

for i, v in enumerate(vs): 
    ax.plot(mean['Time'], -mean[v] / float(v)**2, c=colors[i])
    ax.plot(mean['Time'][mean['Time'] > 10**4], theoreticalMean(float(v), mean['Time'][mean['Time'] > 10**4], step_size) / float(v)**2, ls='--', c=colors[i])

fig.savefig("Mean.png")

''' Make the Variance Plot '''
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([min(var['Time']), max(var['Time'])])

for i, v in enumerate(vs): 
    ax.plot(var['Time'], var[v], c=colors[i])

xvals = np.array([10**3, 10**5])
ax.plot(xvals, xvals**(-1.) / 100, ls='--', c='k', label=r'$t^{-1}$')
ax.plot(xvals, xvals**(-1/2) / 200, ls='-.', c='k', label=r'$t^{-2/3}$')

ax.legend()
fig.savefig("Var.png")

fig, ax = plt.subplots()
final_var = np.array([var[v].values[-1] for v in vs])
ax.set_xscale("log")
ax.set_yscale("log")

ax.scatter(np.array(vs).astype(float), final_var)

fig.savefig("VDependence.png")