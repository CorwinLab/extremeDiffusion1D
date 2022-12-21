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
alpha = 0.5

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Min}})$")
ax.set_xlim([0.5, 750])
ax.set_ylim([10**-2, 10**12])
axin1 = ax.inset_axes([0.55, 0.1, 0.4, 0.4])
axin1.set_yscale("log")
axin1.set_xscale("log")
axin1.set_xlim([1, max(max_dists)])
axin1.set_ylim([10**-2, 10**11])
axin1.set_yticklabels([])
axin1.set_yticks([])
axin1.set_xticklabels([])
axin1.set_xticks([])
axin1.set_xlabel(r"$L$")

end_coord = (2000, 1)
start_coord = (20, 10**8)
axin1.annotate(
    "",
    xy=end_coord,
    xytext=start_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=25,
        headwidth=60,
        headlength=30,
        alpha=0.3,
    ),
    zorder=0,
)
for i, Nexp in enumerate(Ns): 
    dir =  f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    
    N = float(f'1e{Nexp}')
    logN = np.log(N)
    d = np.geomspace(1, 750*logN, num=100)
    var_theory = variance(d, N) + sam_variance_theory(d, N)

    ax.plot(df['Distance'] / logN, df['Variance'], c=colors[i], alpha=alpha, label=Nlabels[i])
    ax.fill_between(df['Distance'] / logN, df['Variance'] - np.sqrt(df['Forth Moment']), df['Variance'] + np.sqrt(df['Forth Moment']), color=colors[i], alpha=alpha/2, edgecolor=None)
    ax.plot(d / logN, var_theory, ls='--', c=colors[i])

    axin1.plot(df['Distance'], df['Variance'], c=colors[i], alpha=alpha, label=Nlabels[i])
    axin1.fill_between(df['Distance'], df['Variance'] - np.sqrt(df['Forth Moment']), df['Variance'] + np.sqrt(df['Forth Moment']), color=colors[i], alpha=alpha/2, edgecolor=None)
    axin1.plot(d, var_theory, ls='--', c=colors[i])

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
