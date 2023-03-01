import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance
from matplotlib.colors import LinearSegmentedColormap
from theory import log_moving_average, log_moving_average_error
from numericalFPT import getNParticleMeanVar
plt.rcParams.update({'font.size': 12})

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Min}})-\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}})$")

colors_used = []

for i, Nexp in enumerate(Ns):
    #cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}' 
    talapas_fixed_dir = f"/home/jacob/Desktop/talapasMount/JacobData/CleanData/FPTCDFPaperFixed/{Nexp}"
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(talapas_fixed_dir, 'MeanVariance.csv')
    print(cdf_file)
    cdf_df = pd.read_csv(cdf_file)
    cdf_df = cdf_df[cdf_df['Distance'] <= 750 * logN]
    
    var_theory = variance(cdf_df['Distance'].values, N)
    ax.plot(cdf_df['Distance'] / logN, var_theory, c=colors[i], ls='--', alpha=0.5)
    
    max_file = os.path.join(dir, 'MeanVariance.csv')
    max_df = pd.read_csv(max_file)
    assert np.array_equal(max_df['Distance'].values, cdf_df['Distance'].values), f"N={N}: {len(max_df['Distance'])}, {len(cdf_df['Distance'])}"

    decade_scaling = 2
    dist_new, env_var = log_moving_average(max_df['Distance'], max_df['Variance'] - cdf_df['Sampling Variance'], 10 ** (1/decade_scaling))
    dist_new, err = log_moving_average_error(max_df['Distance'], max_df['Forth Moment'] + cdf_df['Var Sampling Variance'], 10 ** ( 1/decade_scaling))
    ax.scatter(dist_new / logN, env_var, label=Nlabels[i], color=colors[i], s=2)
    colors_used.append(colors[i])

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors_used,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

ax.set_xlim([0.5, 750])
fig.savefig("EnvironmentalVarianceRecovery.pdf", bbox_inches='tight')