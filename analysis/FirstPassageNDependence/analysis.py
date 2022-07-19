import numpy as np
import npquad 
from matplotlib import pyplot as plt 
import glob

dir = "/home/jacob/Desktop/talapasMount/JacobData/NFirstPassCDF/F*.txt"
files = glob.glob(dir)

data_sum = None 
data_squared = None
number_of_files = 0

for f in files:
    try: 
        data = np.loadtxt(f, delimiter=',', skiprows=1) #columns are: (N_exp, mean, var, quantile)
    except: 
        continue 
    if data.ndim == 1: 
        continue

    if data[-1, 0] != 300:
        continue 

    if data_sum is None: 
        data_sum = data
        data_squared = data**2
    else: 
        data_sum += data
        data_squared += data**2

    number_of_files += 1

mean = data_sum / number_of_files
var_quantile = data_squared[:, 3] / number_of_files - mean[:, 3]**2
Nquads = [np.quad(f"1e{Nexp}") for Nexp in mean[:, 0]]
logN = np.log(Nquads).astype(float)

fig, ax = plt.subplots()
ax.scatter(logN, mean[:, 2], c='k')
ax.plot(logN, 1/logN**4 * 10**11, label=r'$\log(N)^{-4}$')
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{Sam})$")
ax.legend()
fig.savefig("VarianceSam.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.scatter(logN, var_quantile, c='k')
ax.plot(logN, 1/logN**(7/3) * 10**8, label=r'$\log(N)^{-(7/3)}$')
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{Env})$")
ax.legend()
fig.savefig("VarianceEnv.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.scatter(logN, mean[:, 2], c='k', label=r'$\mathrm{Var}(\tau_{Sam})$')
ax.plot(logN, 1/logN**4 * 10**11, c='r', label=r'$\log(N)^{-4}$')
ax.scatter(logN, var_quantile, c='b', label=r'$\mathrm{Var}(\tau_{Env})$', marker='^')
ax.plot(logN, 1/logN**(2) * 10**8 / 6, c='orange', label=r'$\log(N)^{-(2)}$')
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\log(N)$")
ax.set_ylabel(r"\mathrm{Variance}")
ax.legend()
fig.savefig("Variance.pdf", bbox_inches='tight')