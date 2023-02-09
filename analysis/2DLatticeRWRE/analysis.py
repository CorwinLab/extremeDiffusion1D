import glob
import sys
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("../../dataAnalysis")
from FPTDataAnalysis import calculateMeanVarCDF

files = glob.glob("/home/jacob/Desktop/talapasMount/JacobData/2DLatticeRWREHigherFreq/F*.txt")
max_dist = 215

df, num = calculateMeanVarCDF(files, max_dist, verbose=False)
print(df)
logN = np.log(1e24)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^N)$")
ax.plot(df['Distance'] / logN, df['Mean Quantile'])

xvals = np.array([3, 7])
ax.plot(xvals, xvals**2*50, c='k', ls='--', label=r'$L^2$')

xvals = np.array([0.1, 0.7])
ax.plot(xvals, xvals*50, c='grey', ls='--', label=r'$L$')

ax.set_xlim([min(df['Distance']) / logN, max(df['Distance'])/logN])
ax.legend()
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel("Variance")
ax.plot(df['Distance'] / logN, df['Env Variance'], label=r"$\mathrm{Var}(\mathrm{Env}_t^N)$")

ax.plot(df['Distance'] / logN, df['Sampling Variance'], label=r"$\mathrm{Var}(\mathrm{Sam}_t^N)$")

xvals = np.array([2, 3])
ax.plot(xvals, xvals**4 * 2, c='k', ls='--', label=r'$L^4$')

print(xvals**(2/3))
ax.plot(xvals, xvals**(5/3) * 3, c='r', ls='--', label=r'$L^{5/3}$')

ax.set_xlim([min(df['Distance']) / logN, max(df['Distance'])/logN])
ax.set_ylim([10**-2, 10**3])
ax.legend()
fig.savefig("Variance.png", bbox_inches='tight')