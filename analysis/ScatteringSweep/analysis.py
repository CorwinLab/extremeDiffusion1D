import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")
from theory import loglog_moving_average

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweep'
betas = os.listdir(dir)
betas.sort()

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Mean(Env)")
ax.set_xlim([1, 10000])

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("t")
ax2.set_ylabel("Var(Env)")
ax2.set_xlim([1, 10000])

for b in betas: 
    mean = np.loadtxt(os.path.join(dir, b, "Mean.txt"))
    var = np.loadtxt(os.path.join(dir, b, "Var.txt"))
    time = np.loadtxt(os.path.join(dir, b, "Time.txt"))

    ax.plot(time, mean, label=fr'$\beta={b}$')
    if float(b) == 10:
        time, var = loglog_moving_average(time, var, window_size=10**(1/25))

    ax2.plot(time, var, label=fr'$\beta={b}$')

ax.legend()
ax2.legend()
fig.savefig("Mean.pdf", bbox_inches='tight')
fig2.savefig("Var.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_xlim([1, 10000])
ax.set_yscale("log")

for b in betas:
    if float(b) == 0:
        continue

    mean = np.loadtxt(os.path.join(dir, b, "Mean.txt"))
    var = np.loadtxt(os.path.join(dir, b, "Var.txt"))
    time = np.loadtxt(os.path.join(dir, b, "Time.txt"))

    time, var = loglog_moving_average(time, var, window_size=10**(1/25))

    ax.plot(time, var - 1/float(b), label=fr'$\beta={b}$')

xvals = np.array([100, 8*10**3])
yvals = 5*xvals**(-1/2)
ax.plot(xvals, yvals, ls='--', c='k', label=r'$\sqrt{t}$')

yvals = xvals**(-1/3)
ax.plot(xvals, yvals, ls='--', c='m', label=r'$t^{1/3}$')
ax.set_xlabel("t")
ax.set_ylabel(r"Var(Env) - $1/\beta$")
ax.legend()
fig.savefig("VarPowerLaw.pdf", bbox_inches='tight')