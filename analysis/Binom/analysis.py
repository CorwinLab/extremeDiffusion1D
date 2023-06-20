import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import os 

d_smaller_v = '/home/jacob/Desktop/talapasMount/JacobData/BinomV001/'
dir = '/home/jacob/Desktop/talapasMount/JacobData/Binom/'

mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

mean_smaller_v = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/BinomV001/Mean.txt')
var_smaller_v = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/BinomV001/Var.txt')

N = 1_000_000_000_000

# Plot Mean Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Mean(Env)")
ax.set_xlabel("t / log(N)")
ax.plot(mean['Time'] / np.log(N), mean['Quantile'])
ax.plot(mean_smaller_v['Time'] / np.log(N), mean_smaller_v['Quantile'])

fig.savefig("QuantileMean.pdf", bbox_inches='tight')

# Plot Var Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Var(Env)")
ax.set_xlabel("t")
ax.plot(var['Time'] / np.log(N), var['Quantile'])
ax.plot(var_smaller_v['Time'] / np.log(N), var_smaller_v['Quantile'])

fig.savefig("QuantileVar.pdf", bbox_inches='tight')

# Plot Mean Probability
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$-\mathrm{Mean}(\log(P(X=vt^{3/4}, t)))$")
ax.set_xlabel("t / log(N)")
ax.plot(mean['Time'] / np.log(N), -mean['Probability'])
ax.plot(mean_smaller_v['Time'] / np.log(N), -mean_smaller_v['Probability'])
xvals = np.array([10**2, 10**3])
ax.plot(xvals, xvals**(1/4) * 5, ls='--', c='k')
fig.savefig("ProbabilityMean.pdf", bbox_inches='tight')

# Plot Var Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_ylabel(r"$\mathrm{Var}(\log(P(X=vt^{3/4}, t)))$")
ax.set_xlabel("t")
ax.plot(var['Time'] / np.log(N), var['Probability'])
ax.plot(var_smaller_v['Time'] / np.log(N), var_smaller_v['Probability'])

fig.savefig("ProbabilityVar.pdf", bbox_inches='tight')