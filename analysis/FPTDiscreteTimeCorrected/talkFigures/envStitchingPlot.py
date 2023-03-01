import sys 
sys.path.append("../../../dataAnalysis")
from fptTheory import variance, var_power_long, var_short
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os 
import pandas as pd
plt.rcParams.update({'font.size': 15})

alpha = 0.75

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.set_xlim([0.5, 750])
ax.set_ylim([10**-3, 10**12])
Nexp = 12
cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
N = float(f"1e{Nexp}")
logN = np.log(N)
cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], c='tab:blue', alpha=alpha, label=r'$\mathrm{Var}(\tau_{\mathrm{Env}})$')
theory_distances = cdf_df[cdf_df['Distance'] >= logN]['Distance'].values
long = var_power_long(theory_distances, N)
short = var_short(theory_distances, N)
var = variance(theory_distances, N)
ax.plot(np.insert(theory_distances / logN, 0, 0.99), np.insert(short, 0, 0), ls='--', c='tab:orange')#, label=r'$V_1(L, N)$')
ax.plot(theory_distances / logN, long, ls='--', c='m')#, label=r'$V_2(L, N)$')
#ax.plot(theory_distances / logN, var, ls='--', c='k', zorder=-1)
#ax.vlines(logN**(3/2) / logN, 10**-1, 10**10, ls='--', color='k')
#ax.annotate(r'$L=\log(N)^{3/2}$', (3.5, 10**6), rotation=90)

xvals = np.array([100, 600])
ax.plot(xvals, xvals ** (8/3), ls='--', c='k')#, label=r'$L^{8/3}$')
ax.plot(xvals, xvals ** (3)*4, ls='--', c='k')#, label=r'$L^{3}$')

ax.annotate(r'$L^{8/3}$', xy=(250, 10**5*2), rotation=25, fontsize=12)
ax.annotate(r'$L^3$', xy=(200, 10**8), rotation=30, fontsize=12)

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=['tab:blue', 'k', 'k'],
    handlelength=0,
    handletextpad=0,
    fontsize=12,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("EnvironmentalStitching.svg", bbox_inches='tight')