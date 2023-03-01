import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
sys.path.append("../../dataAnalysis")
from theory import log_moving_average
from fptTheory import variance

plt.rcParams.update({'font.size': 12})

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12, 28]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper' # For exponential approximation
talapas_dir = "/home/jacob/Desktop/talapasMount/JacobData/CleanData/FPTCDFPaperFixed" # No exponential approximation
dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/'

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlim([0.5, 750])
ax.set_xlabel(r"$L/\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{No Approx}}(\tau_{\mathrm{Sam}}) - \mathrm{Var}^{\mathrm{Exp Approx}}(\tau_{\mathrm{Sam}})$")
for i, N in enumerate(Ns[:-1]):
    logN = np.log(float(f"1e{N}"))
    path = os.path.join(talapas_dir, str(N), "MeanVariance.csv")
    fixed_df = pd.read_csv(path)
    print(f"Read: {path}")

    path = os.path.join(cdf_dir_specific, str(N), "MeanVariance.csv")
    incorrect_df = pd.read_csv(path)
    print(f"Read: {path}")

    var_diff = fixed_df['Sampling Variance'].values - incorrect_df['Sampling Variance'].values     
    decade_scaling = 10
    dist_new, env_var = log_moving_average(fixed_df['Distance'], var_diff, 10 ** (1/decade_scaling))
    ax.scatter(dist_new, env_var, color=colors[i], label=Nlabels[i])

ax.legend()
fig.savefig("ExponentialApproxDiff.pdf", bbox_inches='tight')