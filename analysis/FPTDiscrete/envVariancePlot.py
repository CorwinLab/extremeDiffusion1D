import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance
from matplotlib.colors import LinearSegmentedColormap
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
ax.set_ylim([10**-2, 10**10])

for i, Nexp in enumerate(Ns):
    cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    var_theory = variance(cdf_df['Distance'].values, N)
    ax.plot(cdf_df['Distance'] / logN, var_theory, c=colors[i], ls='--')
    ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], label=Nlabels[i], c=colors[i], alpha=alpha)

xvals = np.array([100, 600])
ax.plot(xvals, 10 * xvals ** 3, c='k', ls='--', label=r'$L^3$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors + ['k'],
    handlelength=0,
    handletextpad=0,
)
ax.set_ylim([10**-3, 10**12])
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("EnvironmentalVariance.pdf", bbox_inches='tight')