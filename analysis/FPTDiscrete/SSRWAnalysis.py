import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')
N=1e12
logN = np.log2(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlabel(r"$L/\log_2(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{SSRW}})$")
ax.set_xlim([0.5, 500])
xvals = np.array([75, 300])
ax.plot(df['Distance'] / logN, df['Variance'])
ax.plot(df['Distance'] / logN, df['Variance'] - np.pi**2 / 24 * df['Distance'] ** 4 / np.log(N)**4, label=r"$\mathrm{Var}(\tau_{\mathrm{SSRW}}) - \frac{\pi^2 L^4}{24 \log(N)^4}$")
ax.plot(xvals, xvals**4 / 2, ls='--', c='k', label=r'$\pm L^4$')
ax.plot(xvals, -xvals**4 / 2, ls='--', c='k')
ax.legend()
fig.savefig("SSRWVar.pdf", bbox_inches='tight')