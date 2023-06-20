import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
import os

dir = '/home/jacob/Desktop/talapasMount/JacobData/UniformRWMultSteps/'
step_size = os.listdir(dir)
step_size.sort(key=float)

colors = ['r', 'b', 'g']

def theoreticalMean(v, time, step_size):
    sigma2 = 1/9 * step_size * (step_size + 2)
    return v**2 * np.sqrt(time) / 2 / sigma2 

'''Plot the Mean'''
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([0.75, 3*10**2])
ax.set_xlim([1, 10**5])
ax.set_xlabel("t")
ax.set_ylabel(r"$\mathrm{Mean}(\ln(P(X > vt^{3/4}, t)))$")

for i, d in enumerate(step_size): 
    mean_file = os.path.join(dir, d, 'Mean.txt')
    var_file = os.path.join(dir, d, 'Var.txt')

    mean = pd.read_csv(mean_file)
    var = pd.read_csv(var_file)

    ax.plot(mean['Time'], -mean['1'], label=f'm={d}', color=colors[i])
    ax.plot(mean['Time'], theoreticalMean(1, mean['Time'], float(d)), ls='--', color=colors[i])
    
ax.legend()
fig.savefig("MeanLogProb.png")

''' Plot the variance '''

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1, 10**5])
ax.set_xlabel("t")
ax.set_ylabel(r"$\mathrm{Var}(\ln(P(X > vt^{3/4}, t)))$")

for i, d in enumerate(step_size):
    mean_file = os.path.join(dir, d, 'Mean.txt')
    var_file = os.path.join(dir, d, 'Var.txt')

    mean = pd.read_csv(mean_file)
    var = pd.read_csv(var_file)

    ax.plot(var['Time'], var['1'], label=f'm={d}', color=colors[i])

xvals = np.array([10**3, 10**5])
ax.plot(xvals, xvals**(-1/2), ls='--', c='k')
ax.plot(xvals, xvals**(-1/2) / 500, ls='--', c='k', label=r'$t^{-1/2}$')
ax.legend()

fig.savefig("VarLogProb.png")

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\frac{1}{9}m(m+2)$")
ax.set_ylabel(r"$\lim_{t\rightarrow\infty} \quad \mathrm{Var}(\ln(P(X > vt^{3/4}, t)))$")

for d in step_size:
    var_file = os.path.join(dir, d, 'Var.txt')
    var = pd.read_csv(var_file)
    m = float(d)
    ax.scatter(1/9 * m * (m+2), var['1'].values[-1], c='k')

xvals = np.array([0.5, 2*10**1]).astype(float)
ax.plot(xvals, xvals**(-2) / 500, ls='--', c='k', label=r'$\left(\frac{1}{9}m(m+2)\right)^{-2}$')
ax.legend()

fig.savefig("MDependence.png", bbox_inches='tight')