import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory
from matplotlib.colors import LinearSegmentedColormap
from theory import log_moving_average, log_moving_average_error
from scipy.special import lambertw
plt.rcParams.update({'font.size': 12})

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12, 28]

gamma = np.euler_gamma

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Min}})-\mathrm{Var}^{\mathrm{Asy}}(\tau_{\mathrm{Sam}})$")

def moment_function(L, N, m):
    beta = 1/L**2 * lambertw(2 / np.pi * N**2)
    C = L**2 / 2
    D1 = - 1/L**2
    theta=1/2 
    return beta ** m * (1 + m*gamma/C/beta + (m*(m-1)*(6*gamma**2 + np.pi**2) - 12 * m * theta * gamma + 12 * m * C * D1)/(12 * C**2 * beta**2))

colors_used = []

for i, Nexp in enumerate(Ns):
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    
    max_file = os.path.join(dir, 'MeanVariance.csv')
    max_df = pd.read_csv(max_file)

    decade_scaling = 15
    #sam_variance = sam_variance_theory(max_df['Distance'], N)
    sam_variance = moment_function(max_df['Distance'].values, N, -2) - moment_function(max_df['Distance'].values, N, -1)**2
    dist_new, env_var = log_moving_average(max_df['Distance'], max_df['Variance'] - sam_variance, 10 ** (1/decade_scaling))
    dist_new, env_error = log_moving_average_error(max_df['Distance'], max_df['Forth Moment'], 10 ** (1/decade_scaling))
    ax.scatter(dist_new / logN, env_var, label=Nlabels[i], color=colors[i], s=3, alpha=0.75)
    var_theory = variance(max_df['Distance'].values, N)
    ax.plot(max_df['Distance'] / logN, var_theory, c=colors[i], ls='--', alpha=0.5)
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
fig.savefig("EnvironmentalVarianceTheory.pdf", bbox_inches='tight')