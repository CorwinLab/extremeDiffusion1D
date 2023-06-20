import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")
from theory import log_moving_average

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweepLonger'
betas = os.listdir(dir)
betas.sort()
decade_scaling = 10
lw = 1

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Mean(Env)")
ax.set_xlim([1, 100000])

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("t")
ax2.set_ylabel(r"$\left(\frac{1-4\sigma^2_\xi}{8\sigma^2_\xi}\right)\mathrm{Var}(\mathrm{Env})$")
ax2.set_xlim([2, 10000])

# Plot the beta distribution values
for b in betas: 
    if float(b) not in [0.1, 0.01, 1]:
        continue
    mean = np.loadtxt(os.path.join(dir, b, "Mean.txt"))
    var = np.loadtxt(os.path.join(dir, b, "Var.txt"))
    time = np.loadtxt(os.path.join(dir, b, "Time.txt"))
    ax.plot(time, mean, label=fr'$\beta={b}$')

    time, var = log_moving_average(time, var, 10**(1/decade_scaling))
    ax2.plot(time, var * float(b), label=fr'$\beta={b}$', alpha=0.5, lw=lw)

# Plot delta distribution variance
var = np.loadtxt(os.path.join('/home/jacob/Desktop/talapasMount/JacobData/ScatteringDeltaDist/Var.txt'))
time  = np.loadtxt(os.path.join('/home/jacob/Desktop/talapasMount/JacobData/ScatteringDeltaDist/Time.txt'))

time, var = log_moving_average(time, var, 10**(1/decade_scaling))
ax2.plot(time, var, label='Delta Dist', alpha=0.5, lw=lw)

# Plot different N dependence
dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringNSweep'
Nexps = os.listdir(dir)
Nexps.sort()
b = 1

for Nexp in Nexps:
    var = np.loadtxt(os.path.join(dir, Nexp, "Var.txt"))
    time = np.loadtxt(os.path.join(dir, Nexp, "Time.txt"))
    N = float(f"1e{Nexp}")
    time, var = log_moving_average(time, var, 10**(1/decade_scaling))
    ax2.plot(time, var * float(b), label=fr'$N=1e{Nexp}, \beta={b }$', alpha=0.5, lw=lw)

ax.legend()
ax2.legend()
ax2.set_ylim([10**-1, 10])
ax2.set_xlim([10, 10**4])
#ax2.hlines(1, 10, 10**4, color='k', ls='--')

fig.savefig("Mean.pdf", bbox_inches='tight')
fig2.savefig("Var.pdf", bbox_inches='tight')