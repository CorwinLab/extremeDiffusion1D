import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance, var_power_long, var_short
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os 
import pandas as pd
plt.rcParams.update({'font.size': 15})

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12, 28]
max_dists = [1725, 3452, 8630, 20721, np.log(1e28) * 750]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.set_xlim([0.5, 10**3])
ax.set_ylim([10**-1, 10**10])
Nexp = 12
cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
N = float(f"1e{Nexp}")
logN = np.log(N)
cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], c=colors[-2], alpha=alpha)
theory_distances = cdf_df[cdf_df['Distance'] >= logN]['Distance'].values
long = var_power_long(theory_distances, N)
short = var_short(theory_distances, N)
var = variance(theory_distances, N)
ax.plot(theory_distances / logN, short, ls='--', c='tab:red', label=r'$V_1(L, N)$')
ax.plot(theory_distances / logN, long, ls='--', c='tab:green', label=r'$V_2(L, N)$')
ax.plot(theory_distances / logN, var, ls='--', c='k', zorder=-1)
ax.vlines(logN**(3/2) / logN, 10**-1, 10**10, ls='--', color='k')
ax.annotate(r'$L=\log(N)^{3/2}$', (3.5, 10**6), rotation=90)

leg = ax.legend(
    loc="lower right",
    framealpha=0,
    labelcolor=['tab:red', 'tab:green'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

'''This isn't helpful
axins = ax.inset_axes([0.53, 0.05, 0.45, 0.45])
axins.plot(theory_distances / logN, short, ls='--', c='tab:red')
axins.plot(theory_distances / logN, long, ls='--', c='tab:green')
axins.plot(theory_distances / logN, var, ls='--', c='k', zorder=-1)
axins.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], c=colors[-2], alpha=alpha)
mult=2
axins.set_xlim([5, 6])
axins.set_ylim([2*10 ** 2, 3 * 10 ** 2])
axins.set_xscale("log")
axins.set_yscale("log")
axins.xaxis.set_ticklabels([])
axins.yaxis.set_ticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
'''
fig.savefig("EnvironmentalStitching.pdf", bbox_inches='tight')