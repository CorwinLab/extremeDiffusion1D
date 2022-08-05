import numpy as np
import pandas as pd
import npquad 
from matplotlib import pyplot as plt 
import glob 
import os

directory = "/home/jacob/Desktop/talapasMount/JacobData/MultipleNFirstPassCDF"

figvar, axvar = plt.subplots()
figquant, axquant = plt.subplots()

for folder in os.listdir(directory):
    files = glob.glob(os.path.join(directory, folder, 'F*.txt'))
    number_of_non_empty_files = 0
    variance_sum = None
    quantile_sum = None 
    quantile_sum_squared = None
    for f in files:
        # if file is empty just continue
        try: 
            df = pd.read_csv(f, delimiter=',')
        except pd.errors.EmptyDataError:
            continue

        number_of_non_empty_files += 1
        if variance_sum is None: 
            variance_sum = df['var'].values 
        else: 
            variance_sum += df['var'].values
        if quantile_sum is None: 
            quantile_sum = df['quantile'].values 
        else: 
            quantile_sum += df['quantile'].values
        if quantile_sum_squared is None:
            quantile_sum_squared = df['quantile'].values ** 2
        else: 
            quantile_sum_squared += df['quantile'].values ** 2
    
    x = int(folder)
    quantile_var = quantile_sum_squared / number_of_non_empty_files - (quantile_sum / number_of_non_empty_files) ** 2
    variance_mean = variance_sum / number_of_non_empty_files
    N = np.log(df['N'].values).astype(float)

    axvar.scatter(N/x, variance_mean, label=f"x={x}", alpha=0.75)
    axquant.scatter(N/x, quantile_var, label=f"x={x}", alpha=0.75)

xvals = np.array([min(N/x), 1.2])
yvals = xvals ** (-4)

axvar.plot(xvals, yvals, c='k', ls='--', label=r'$(\log(N))^{-4}$')
axvar.set_xlabel(r"$\log(N) / x$")
axvar.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}})$")
axvar.set_yscale("log")
axvar.set_xscale("log")
axvar.set_ylim([10**-2, 2*10**6])
axvar.set_xlim([0, 1.2])
axvar.legend()
figvar.savefig("VarSam.pdf", bbox_inches='tight')

power = -2.5
power2 = -2
yvals_quant = xvals ** (power)
yvals_quant2 = xvals ** (power2)
axquant.plot(xvals, yvals_quant * 10, c='k', ls='--', label=r"$(\log(N))^{-2.5}$")
axquant.plot(xvals,yvals_quant2 * 10, c='k', ls='-.', label=r"$\log(N)^{-2}$")
axquant.set_xlabel(r"$log(N) / x$")
axquant.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
axquant.set_yscale("log")
axquant.set_xscale("log")
axquant.set_ylim([10**-1, 2*10**5])
axquant.set_xlim([0, 1.2])
axquant.legend()
figquant.savefig("VarEnv.pdf", bbox_inches='tight')