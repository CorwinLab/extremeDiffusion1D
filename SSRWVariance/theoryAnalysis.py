import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../dataAnalysis")
from numericalFPT import getNParticleMeanVar

df = pd.read_csv("MeanVar12.txt")

N = 1e12
logN = np.log2(N)

num_mean, num_var = getNParticleMeanVar(df['Position'].values, N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log_2(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{SSRW}})$")
ax.set_xlim([0.8, 500])
ax.set_ylim([10**-3, 10**12])
ax.scatter(df['Position'].values / logN, df['Variance'].values, label='Master Equation')
ax.plot(df['Position'].values / logN, num_var, label='Continuous Approx', c='k')
ax.legend()
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log_2(N)$")
ax.set_ylabel("|Residual|")
ax.set_xlim([0.5, 550])
ax.set_ylim([0.5, 10**6])
ax.scatter(df['Position'].values / logN, np.abs(df['Variance'].values - num_var))

xvals = np.array([10, 300])
ax.plot(xvals, xvals**2 * 2, ls='--', c='k', label=r'$L^2$')

ax.legend()
fig.savefig("Residual.png", bbox_inches='tight')

einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log_2(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{SSRW}})$")
ax.set_xlim([0.8, 500])
ax.set_ylim([10**-3, 10**12])
ax.plot(einstein_df['Distance'].values / logN, einstein_df['Variance'], label='Discrete Data', ls='--')
ax.scatter(df['Position'].values / logN, df['Variance'].values, label='Master Equation')
ax.legend()
fig.savefig("NumericalVariance.png", bbox_inches='tight')

num_mean, num_var = getNParticleMeanVar(einstein_df['Distance'].values, N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log_2(N)$")
ax.set_ylabel("|Residual|")
ax.set_xlim([0.5, 550])
ax.plot(einstein_df['Distance'].values / logN, np.abs(einstein_df['Variance'] - num_var))
xvals = np.array([10, 300])
ax.plot(xvals, xvals ** 4 / 2, c='k', ls='--', label=r'$L^4$')
ax.legend()
fig.savefig("NumericalResidual.png", bbox_inches='tight')

xvals = np.geomspace(10**(-24), 1, num=500)
print(1-(1-xvals)**N)