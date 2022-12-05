import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory
from theory import log_moving_average
from pyDiffusion.quadMath import prettifyQuad
plt.rcParams.update({'font.size': 15})

def delta(L, N):
    logN = np.log(N)
    prefactor = 1 / (2 * logN / L**2 - 2 *logN**2 / L**2 - 4 * logN**4/L**4)
    return prefactor * (-logN**3/3/L**2 - np.log(np.sqrt(16*np.pi*logN**3/L**2)) +np.log(2) + np.log(2*logN/L)-2*logN**3/3/L**2)

def different_sam_theory(x, N):
    logN = np.log(N) 
    t = x**2 / 2 / logN + delta(x, N)
    return np.pi**2 / 6 / (x**2/2/t**2 + 1/2/t)**2

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Variance}$")
ax.set_xlim([0.5, 750])

max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
einstein_color = "tab:purple"
alpha = 0.75

Nexp = 12
dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected/{Nexp}/'
meanFile = os.path.join(dir, 'MeanVariance.csv')
max_df = pd.read_csv(meanFile)
N = float(f'1e{Nexp}')
logN = np.log(N)
ax.set_title(f"N=1e12")

cdf_dir = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}/'
cdf_file = os.path.join(cdf_dir, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)

einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')

env_theory = variance(cdf_df['Distance'].values, N)
sam_theory = sam_variance_theory(cdf_df['Distance'].values, N)
new_row = [np.log2(N), 0]
einstein_theoretical_data = np.loadtxt("etheoretical12.txt")
einstein_theoretical_data = np.vstack([new_row, einstein_theoretical_data])
ax.errorbar(max_df['Distance'] / logN, max_df['Variance'], yerr=np.sqrt(max_df['Forth Moment']), label=r'$\mathrm{Var}(\tau_{\mathrm{Min}})$', c=max_color, alpha=alpha)
ax.errorbar(cdf_df['Distance'] / logN, cdf_df['Sampling Variance'], yerr=np.sqrt(cdf_df['Var Sampling Variance']), label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$', c=gumbel_color, alpha=alpha, lw=0.5)
ax.errorbar(cdf_df['Distance'] / logN, cdf_df['Env Variance'], yerr=np.sqrt(cdf_df['Var Env Variance']), label=r'$\mathrm{Var}(\tau_{\mathrm{Env}})$', c=quantile_color, alpha=alpha)
ax.plot(einstein_df['Distance'] / logN, einstein_df['Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{SSRW}})$', c=einstein_color, alpha=alpha)

ax.plot(einstein_theoretical_data[:, 0] / logN, einstein_theoretical_data[:, 1], ls='--', c=einstein_color)
ax.plot(cdf_df['Distance'] / logN, env_theory, ls='--', c=quantile_color)
ax.plot(cdf_df['Distance'] / logN, sam_theory, ls='--', c=gumbel_color)
ax.plot(cdf_df['Distance'] / logN, env_theory + sam_theory, ls='--', c=max_color)

xvals = np.array([100, 600])
ax.plot(xvals, xvals ** 4, label=r'$L^{4}$', c='k', ls='--')
ax.plot(xvals, xvals ** 3, label=r'$L^{3}$', c='k', ls='--')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    #labelcolor=[max_color, gumbel_color, quantile_color, einstein_color, 'k', 'k'],
    labelcolor=[einstein_color, 'k', 'k', max_color, gumbel_color, quantile_color],
    handlelength=0,
    handletextpad=0,
    fontsize=12,
)
ax.set_ylim([10**-3, 10**12])
for item in leg.legendHandles:
    item.set_visible(False)
fig.savefig("CompleteVariance.pdf", bbox_inches='tight')