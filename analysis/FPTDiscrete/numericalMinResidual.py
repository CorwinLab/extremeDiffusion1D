import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from matplotlib.colors import LinearSegmentedColormap
from theory import log_moving_average
plt.rcParams.update({'font.size': 12})

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
ax.set_yscale("symlog")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Min}})-(\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}}) + \mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Env}}))}$")

for i, Nexp in enumerate(Ns):
    print(Nexp)
    cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper/{Nexp}/'

    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    
    max_file = os.path.join(dir, 'MeanVariance.csv')
    max_df = pd.read_csv(max_file)

    decade_scaling = 25

    max_at_cdf = np.interp(cdf_df['Distance'].values, max_df['Distance'].values, max_df['Variance'])
    dist_new, fractional_error = log_moving_average(cdf_df['Distance'].values, (max_at_cdf - (cdf_df['Sampling Variance'] + cdf_df['Env Variance'])), 10**(1/decade_scaling))
    ax.scatter(dist_new / logN, fractional_error, color=colors[i], label=Nlabels[i], s=3)
    #ax.scatter(cdf_df['Distance'] / logN, (max_at_cdf - (cdf_df['Sampling Variance'] + cdf_df['Env Variance'])) / max_at_cdf * 100, color=colors[i], s=1, label=Nlabels[i])
    #ax.errorbar(cdf_df['Distance'] / logN, (max_at_cdf - (cdf_df['Sampling Variance'] + cdf_df['Env Variance'])), yerr=np.sqrt(fourth_moment), alpha=0.5, lw=0.5, fmt='o', c=colors[i], ms=1)

xvals = np.array([100, 600])
ax.plot(xvals, xvals ** 4 / 200, c='k', ls='--', label=r'$\pm L^4$')
ax.plot(xvals, -xvals ** 4 / 200, c='k', ls='--')
#ax.set_ylim([-100, 100])
leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=['k'] + colors,
    handlelength=0,
    handletextpad=0,
)
ax.set_xlim([0.5, 750])
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("NumericalResidual.pdf", bbox_inches='tight')