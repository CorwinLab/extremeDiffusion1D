import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import sam_variance_theory
from matplotlib.colors import LinearSegmentedColormap
from theory import log_moving_average, log_moving_average_error
plt.rcParams.update({'font.size': 12})

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12, 28]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \ln(N)$")
ax.set_ylabel(r"$|\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}})-\mathrm{Var}^{\mathrm{Asy}}(\tau_{\mathrm{Sam}})|$")
ax.set_ylim([10**-2, 10**11])
colors_used = []

for i, Nexp in enumerate(Ns):
    cdf_dir_specific = f"/home/jacob/Desktop/talapasMount/JacobData/CleanData/FPTCDFPaperFixed/{Nexp}"
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    cdf_df = cdf_df[cdf_df['Distance'] <= 750 * logN]
    sampling_variance = sam_variance_theory(cdf_df['Distance'].values, N)

    decade_scaling = 15
    ax.plot(cdf_df['Distance'] / logN, np.abs(cdf_df['Sampling Variance'] - sampling_variance), 10 ** (1/decade_scaling), c=colors[i])
    colors_used.append(colors[i])


xvals = np.array([100, 600])
ax.plot(xvals, xvals**4 / 4, ls='--', c='k', label=r'$L^{4}$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=['k'] + colors_used,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

ax.set_xlim([0.5, 750])
fig.savefig("SamplingResidual.pdf", bbox_inches='tight')