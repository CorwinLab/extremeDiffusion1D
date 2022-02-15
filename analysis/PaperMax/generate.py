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

fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{max}_t^{(N)}) / \lnN^{2/3}$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

ax2 = fig.add_axes([0.53, 0.21, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax2.set_ylabel(r"$\mathrm{Mean}(\mathrm{max}_t^{(N)})$", labelpad=0, fontsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.set_xlim([10**-3, 5*10**3])
ax2.set_ylim([1, 10**5])

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
    max_df['Mean Max'] = max_df['Mean Max'] * 2

    var_theory = theory.quantileVar(Nquad, max_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))
    mean_theory = theory.quantileMean(Nquad, max_df['time'].values)

    ax.plot(max_df['time'] / logN, (var_theory + theory.gumbel_var(max_df['time'].values, Nquad)) / logN**(ypower), '--', c=colors[i])
    ax.plot(max_df['time'] / logN, max_df['Var Max'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)

    ax2.plot(max_df['time'] / logN, max_df['Mean Max'], c=colors[i], alpha=0.8)
    ax2.plot(max_df['time'] / logN, mean_theory, '--', c=colors[i])

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-4, 2*10**3])
fig.savefig("MaxVar.png", bbox_inches='tight')
