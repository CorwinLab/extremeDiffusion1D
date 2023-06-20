import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from numba import njit
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_var_fit

dir = '/home/jacob/Desktop/talapasMount/JacobData/MultiJumpRWDirichlet/'
mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

def theoreticalMean(t, N, sigma):
    return np.sqrt(-t * np.log(N) / (1/sigma-1/2))

v = 1/2
step_size = 11
width = step_size // 2
xvals = np.arange(-width, width + 1)
sigma = np.sqrt(np.sum(xvals**2 / len(xvals)))

N = 1e12
logN = np.log(N)
gamma = 0.121
q0 = 1/11

# Plot the mean quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel("Mean(Env)")
ax.plot(mean['Time'] / logN, mean['Quantile'])

# Linear 
xvals = np.array([10**-2 * 6, 10**-1 * 2])
ax.plot(xvals, xvals * 100, ls='--', c='r', label=r'$t$')

# Sqrt
xvals = np.array([10**2, 10**3])
ax.plot(xvals, np.sqrt(xvals) * 100, c='k', ls='--', label=r'$\sqrt{t}$')
ax.set_xlim([min(mean['Time'] / logN), max(mean['Time'] / logN)])
# Theory
#ax.plot(mean['Time'] / logN, theoreticalMean(mean['Time'], N, sigma)) # multiplying by a factor of 3 seems to work 
ax.legend()
fig.savefig("QuantileMean.pdf", bbox_inches='tight')

# Plot quantile variance 
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel("Var(Env)")
ax.set_xlim([min(mean['Time'] / logN), max(mean['Time'] / logN)])
ax.set_ylim([10**-1, 5*10**1])

ax.plot(var['Time'] / logN, var['Quantile'])

xvals = np.array([10**2, 10**3])
ax.plot(xvals, np.sqrt(xvals), ls='--', c='k', label=r'$\sqrt{t}$')
ax.legend()
fig.savefig("QuantileVariance.pdf", bbox_inches='tight')

def logGaussian(v, t, sigma):
    return -v**2 * t**(3/2) / 2 / sigma**2 - np.log(np.sqrt(2 * np.pi * sigma**2))

# Plot probability mean
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel(r"$-\mathrm{Mean}(\ln(P(vt^{3/4}, t))$")
ax.plot(mean['Time'], -mean['Probability'], label='Data')

#theory = logGaussian(v, mean['Time'], np.sqrt(10*mean['Time']))
#ax.plot(mean['Time'], -theory, ls='--', c='m', alpha=0.75, label='Mean Field Theory')
ax.legend()
fig.savefig("ProbabilityMean.pdf", bbox_inches='tight')

# Plot probability mean
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel(r"$Var(\ln(P(vt^{3/4}, t))$")
ax.plot(var['Time'], var['Probability'])

theory = KPZ_var_fit((gamma-q0) / (1-gamma) * v**4)
#ax.hlines(theory, 10**4, 10**5, ls='--', color='k')
fig.savefig("ProbabilityVar.pdf", bbox_inches='tight')