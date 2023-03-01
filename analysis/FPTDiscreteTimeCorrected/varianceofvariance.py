import numpy as np
from matplotlib import pyplot as plt
import os 
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']

Ns = [1, 2, 5, 12, 28]
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Ns) / 1) for i in range(len(Ns))]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\sqrt{\mathrm{Var}(\mathrm{Var}(\tau_{\mathrm{Min}}))}$")

for i, Nexp in enumerate(Ns): 
    dir =  f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    var_theory = variance(df['Distance'].values, N)
    ax.plot(df['Distance'] / logN, var_theory, c=colors[i], ls='--', alpha=0.5)
    ax.plot(df['Distance'].values / logN, np.sqrt(df['Forth Moment']), color=colors[i], label=Nlabels[i])

xvals = np.array([100, 600])
yvals = (xvals)**4
ax.plot(xvals, yvals, ls='--', c='k', label=r'$L^8$')

ax.set_xlim([0.1, 750])
ax.legend()
fig.savefig("FourthMoment.pdf", bbox_inches='tight')