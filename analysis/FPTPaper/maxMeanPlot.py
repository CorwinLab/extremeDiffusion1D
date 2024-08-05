import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import mean_theory
import numpy as np
plt.rcParams.update({'font.size': 15})

Nlabels = [r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [2, 5, 12, 28]
max_dists = [3452, 8630, 20721, np.log(1e28) * 750]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.75

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \ln(2N)$")
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Min}_L^N)$")
ax.set_xlim([1/(np.log(1e28) + np.log(2)), 500])
ax.set_ylim([1, 3*10**7])
for i, Nexp in enumerate(Ns): 
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N) + np.log(2)
    mean = mean_theory(df['Distance'].values, N)
    
    ax.plot(df['Distance'] / logN, df['Mean'], c=colors[i], alpha=alpha, label=Nlabels[i])
    #ax.fill_between(df['Distance'] / logN, df['Mean'] - np.sqrt(df['Variance']), df['Mean'] + np.sqrt(df['Variance']), color=colors[i], alpha=alpha/2, edgecolor=None)
    ax.plot(df['Distance'] / logN, mean, c=colors[i], ls='--')

xvals = np.array([100, 450])
ax.plot(xvals, 50*xvals**2, ls='--', c='k', label=r'$L^2$')

xvals = np.array([0.05, 0.5])
ax.plot(xvals, xvals*100, ls='--', c='k', label=r'$L$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors +['k', 'k'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("./Figures/MaxMean.pdf", bbox_inches='tight')
