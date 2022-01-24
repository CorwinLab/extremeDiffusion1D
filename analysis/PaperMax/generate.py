import sys
sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import os

def getLessThanT(time, mean):
    greater = mean >= time-1
    nonzero = np.nonzero(greater)[0][-1]
    return time[nonzero]

db = Database()
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type='Max')
    #db.calculateMeanVar(path, verbose=True)

quantiles = db.N(dir_type='Max')
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\sigma^{2}_{max} / \ln(N)^{2/3}$")

cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
ypower = 2/3
for i, N in enumerate(quantiles):
    _, max_df = db.getMeanVarN(N)

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)
    max_df['Var Max'] = max_df['Var Max'] * 4
    var_theory = theory.quantileVar(Nquad, max_df['time'].values)

    ax.plot(max_df['time'] / logN, (var_theory)/ logN**(2/3) + np.pi**2 / 12 * max_df['time'] / logN / logN**(2/3), '--', c=colors[i])
    ax.plot(max_df['time'] / logN, max_df['Var Max'] / logN**(2/3), label=N, c=colors[i], alpha=0.5)

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-4, 2*10**3])
fig.savefig("MaxVar.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\overline{X_{max}}(N, t)$")

ax2 = fig.add_axes([0.2, 0.57, 0.25, 0.25])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\ln(N)$", fontsize=8, labelpad=0)
ax2.set_ylabel(r"$\tau$", fontsize=8, labelpad=0)
ax2.tick_params(axis='both', which='major', labelsize=6)

cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
logNs = []
t_less_than = []

for i, quantile in enumerate(quantiles):
    N = np.quad(f"1e{quantile}")
    logN = np.log(N).astype(float)
    _, max_df = db.getMeanVarN(quantile)

    max_df['Mean Max'] = max_df['Mean Max'] * 2
    var_theory = theory.quantileMean(N, max_df['time'].values)

    ax.plot(max_df['time'] / logN, max_df['Mean Max'], label=quantile, c=colors[i], alpha=0.8)
    ax.plot(max_df['time'] / logN, var_theory, '--', c=colors[i])

    t_less_than.append(getLessThanT(max_df['time'].values, max_df['Mean Max'].values))
    logNs.append(logN)

ax2.scatter(logNs, t_less_than, c=colors)
ax2.plot(logNs, logNs, c='k', ls='--')
ax.set_xlim([10**-3, 5*10**3])
ax.set_ylim([1, 10**5])
fig.savefig("Mean.png")
