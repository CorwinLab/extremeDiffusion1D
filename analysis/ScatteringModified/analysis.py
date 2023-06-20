import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")
from theory import quantileVar

# Plot different N dependence
dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringModified'
Nexps = os.listdir(dir)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env})$")
ax.set_xlim([2/np.log(1e5), 10000/np.log(1e5)])

for Nexp in Nexps:
    var = np.loadtxt(os.path.join(dir, Nexp, "Var.txt"))
    time = np.loadtxt(os.path.join(dir, Nexp, "Time.txt"))
    N = float(f"1e{Nexp}")

    ax.plot(time / np.log(N), var, label=fr'$N=1e{Nexp}, \beta=1$')

    theoretical_var = quantileVar(N, time)
    ax.plot(time / np.log(N), theoretical_var, ls='--', c='k')
    

fig.savefig("Var.pdf", bbox_inches='tight')