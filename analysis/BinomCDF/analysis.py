import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import os 

dir = '/home/jacob/Desktop/talapasMount/JacobData/BinomCDF/'

mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)


N = 1_000_000_000_000

# Plot Mean Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Mean(Env)")
ax.set_xlabel("t / log(N)")
ax.plot(mean['Time'] / np.log(N), mean['Quantile'])
fig.savefig("QuantileMean.pdf", bbox_inches='tight')

# Plot Var Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Var(Env)")
ax.set_xlabel("t")
ax.plot(var['Time'] / np.log(N), var['Quantile'])
fig.savefig("QuantileVar.pdf", bbox_inches='tight')

# Plot Mean Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\mathrm{Mean}(P(X=vt^{3/4}, t))$")
ax.set_xlabel("t / log(N)")
xvals = np.array([10, 5*10])
#ax.plot(xvals, xvals**(1/8)*5, ls='--', c='k')
ax.plot(mean['Time'] / np.log(N), -mean['Probability'])
fig.savefig("ProbabilityMean.pdf", bbox_inches='tight')

# Plot Var Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\mathrm{Var}(P(X=vt^{3/4}, t))$")
ax.set_xlabel("t")
ax.plot(var['Time'] / np.log(N), var['Probability'])
fig.savefig("ProbabilityVar.pdf", bbox_inches='tight')