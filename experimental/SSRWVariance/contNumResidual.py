import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../dataAnalysis")
from numericalFPT import getNParticleMeanVar

N = 1e12
logN = np.log(N)
df = pd.read_csv("MeanVar12.txt")
mean, var = getNParticleMeanVar(df['Position'].values, N)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel("-Residual")
ax.plot(df['Position'].values / logN, -(df['Variance'].values - var))

xvals = np.array([2, 20])
ax.plot(xvals, xvals**2, ls='--', c='k', label=r'$L^2$')

xvals = np.array([100, 500])
ax.plot(xvals, xvals**4 / 1000 / 5, ls='--', c='r', label=r'$L^4$')
ax.legend()
fig.savefig("NewVariance.png")