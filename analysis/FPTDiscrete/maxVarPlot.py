import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory
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
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Min}})$")
ax.set_xlim([1, max(max_dists)])
ax.set_ylim([10**-2, 10**11])
for i, Nexp in enumerate(Ns): 
    dir =  f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N)

    var_theory = variance(df['Distance'].values, N) + sam_variance_theory(df['Distance'].values, N)

    ax.plot(df['Distance'], df['Variance'], c=colors[i], alpha=alpha, label=Nlabels[i])
    ax.plot(df['Distance'], var_theory, ls='--', c=colors[i])

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("MaxVariance.pdf", bbox_inches='tight')
