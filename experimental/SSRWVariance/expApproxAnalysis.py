import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

exp_df = pd.read_csv("./ExpApprox/MeanVar2.txt")
true_df = pd.read_csv("./NoApprox/MeanVar2.txt")

exp_df = exp_df[:400]
true_df = true_df[:400]

N = 1e5
logN = np.log(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(exp_df['Position'].values / logN, exp_df['Variance'])
ax.plot(true_df['Position'] / logN, true_df['Variance'], ls='--')
ax.set_ylim([10**-2, 10**11])
fig.savefig("ExpApprox.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-5, 10**9])
ax.set_xlim([0.8, 300])
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Exp Approx}}(\tau_{\mathrm{Sam}}) - \mathrm{Var}^{\mathrm{No Approx}}(\tau_{\mathrm{Sam}})$")

for Nexp in [2, 5]:
    exp_df = pd.read_csv(f"./ExpApprox/MeanVar{Nexp}.txt")
    true_df = pd.read_csv(f"./NoApprox/MeanVar{Nexp}.txt")

    N = float(f"1e{Nexp}")
    logN = np.log(N)
    exp_df = exp_df[exp_df['Position'] / logN < 300]
    true_df = true_df[true_df['Position'] / logN < 300]

    ax.plot(exp_df['Position'] / logN, -(true_df['Variance'] - exp_df['Variance']), label=f'1e{Nexp}')

xvals = np.array([10, 600])
ax.plot(xvals, xvals**4 / 10, ls='--', c='k', label=r'$L^4$')
ax.legend()
fig.savefig("ExpDiff.png", bbox_inches='tight')