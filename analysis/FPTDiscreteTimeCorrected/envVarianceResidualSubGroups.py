import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance
from matplotlib.colors import LinearSegmentedColormap
from theory import log_moving_average, log_moving_average_error
plt.rcParams.update({'font.size': 12})

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / 10 / 1) for i in range(10)]
alpha = 0.6
Nexp = 12

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Min}})-\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}})$")

colors_used = []

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")

cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
N = float(f"1e{Nexp}")
logN = np.log(N)
for i in range(10):
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    cdf_df = cdf_df[cdf_df['Distance'] <= 500 * logN]

    max_file = os.path.join(dir, f'MeanVariance{i}.csv')
    max_df = pd.read_csv(max_file)
    assert np.array_equal(max_df['Distance'].values, cdf_df['Distance'].values)
    env_error = max_df['Forth Moment'] + cdf_df['Var Sampling Variance']
    ax2.scatter(max_df['Distance'] / logN, max_df['Forth Moment'] / cdf_df['Var Sampling Variance'], color=colors[i])

    decade_scaling = 15
    dist_new, env_var = log_moving_average(max_df['Distance'], max_df['Variance'] - cdf_df['Sampling Variance'], 10 ** (1/decade_scaling))
    dist_new, env_error = log_moving_average_error(max_df['Distance'], env_error, 10 ** (1/decade_scaling))
    ax.scatter(dist_new / logN, env_var, color=colors[i], s=1)
    colors_used.append(colors[i])

var_theory = variance(cdf_df['Distance'].values, N)
ax.plot(cdf_df['Distance'] / logN, var_theory, c=colors[i], ls='--', alpha=0.5)

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors_used,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

#ax.set_ylim([10**-3, 10**12])
ax.set_xlim([0.5, 500])
fig.savefig("EnvironmentalVarianceResidual.pdf", bbox_inches='tight')

ax2.set_xlim([1, 500])
ax2.set_ylim([10**-3, 7 * 10**-1])
ax2.set_xlabel(r"$L / \log(N)$")
ax2.set_ylabel(r"$\frac{\delta\tau_{\mathrm{Min}}}{\delta\tau_{\mathrm{Sam}}}$")
fig2.savefig("RelativeVariance.pdf", bbox_inches='tight')