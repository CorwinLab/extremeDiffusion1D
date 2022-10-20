import numpy as np
from matplotlib import pyplot as plt
import glob 
import pandas as pd

dir = '/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassNDependence/F*.txt'
files = glob.glob(dir)
run_again = False

if run_again:
    nFiles = 0
    data_squared = None
    data_sum = None
    for f in files: 
        df = pd.read_csv(f)
        if max(df['N']) != 1e23: 
            continue
        quantile = df['Quantile'].values
        variance = df['Variance'].values
        data = np.array([quantile, variance])
        if data_squared is None:
            data_squared = data ** 2
        else: 
            data_squared += data ** 2
        if data_sum is None: 
            data_sum = data 
        else: 
            data_sum += data

        Ns = df['N']
        distances = df['Distance']
        nFiles += 1

    mean = data_sum / nFiles 
    var = data_squared / nFiles - mean**2

    df_tot = pd.DataFrame()
    df_tot["N"] = Ns
    df_tot["Distance"] = distances 
    df_tot["Mean Quantile"] = mean[0, :]
    df_tot["Quantile Variance"] = var[0, :]
    df_tot["Gumbel Variance"] = mean[1, :]
    df_tot.to_csv("TotalData.csv", index=False)

else:
    df_tot = pd.read_csv("TotalData.csv")

fig, ax = plt.subplots()
ax.scatter(np.log(df_tot["N"]), df_tot["Quantile Variance"])
xvals = np.array([5, 55])
yvals = xvals ** (1/2)
ax.plot(xvals, 3.2 * yvals * 10**5, c='r', ls='--', label=r'$\log(N)^{1/2}$')
ax.set_xlabel("log(N)")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("Variance.pdf", bbox_inches='tight')