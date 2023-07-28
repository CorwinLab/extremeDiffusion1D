import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import sys 
sys.path.append("../dataAnalysis")
from fptTheory import variance

Nexp = 2
talapas_dir = f"/home/jacob/Desktop/talapasMount/JacobData/CleanData/FPTCDFPaperFixed/{Nexp}/MeanVariance.csv" 
RWRE_df = pd.read_csv(talapas_dir)

SSRW_file = f"/home/jacob/Desktop/talapasMount/JacobData/SSRWFPT/{Nexp}/MeanVariance.txt"
SSRW_df = pd.read_csv(SSRW_file)

N = float(f"1e{Nexp}")
logN = np.log(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-3, 10**12])
ax.set_xlim([0.5, 750])
ax.plot(RWRE_df['Distance'] / logN, RWRE_df['Sampling Variance'], label='RWRE')
ax.plot(SSRW_df['Position'] / logN, SSRW_df['Variance'], label='SSRW')
ax.legend()
fig.savefig("RWRESSRWComp.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.5, 750])
ax.set_ylim([10**-3, 10**9])
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}}) - \mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{SSRW}})$")
ax.scatter(RWRE_df['Distance'] / logN, RWRE_df['Sampling Variance'] - SSRW_df['Variance'], label='N=1e2')
xvals = np.array([100, 600])
ax.plot(xvals, xvals**3/ 100, ls='--', c='k', label=r'$L^3$')
ax.plot(SSRW_df['Position'].values / logN, variance(SSRW_df['Position'].values, N), c='r', label=r'$\mathrm{Var}^{\mathrm{Asy}}(\tau_{\mathrm{Env}})$')
ax.legend()
fig.savefig("RWRESSRWDiff.png", bbox_inches='tight')