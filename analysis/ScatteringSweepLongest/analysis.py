import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")
from theory import log_moving_average

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweepLongest/0.01'
b = 0.01

mean = np.loadtxt(os.path.join(dir, "Mean.txt"))
var = np.loadtxt(os.path.join(dir, "Var.txt"))
time = np.loadtxt(os.path.join(dir, "Time.txt"))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Mean(Env)")
ax.set_title(r"$\beta=0.01$")
ax.set_xlim([1, 500_000])
ax.plot(time, mean, label=fr'$\beta={b}$')
fig.savefig("Mean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Var(Env)")
ax.set_title(r"$\beta=0.01$")
ax.set_xlim([1, 500_000])
ax.set_ylim([1, 2*10**2])

#time, var = log_moving_average(time, var, 10**(1/25))
ax.scatter(time, var, label=fr'$\beta={b}$', s=1)

fig.savefig("Var.pdf", bbox_inches='tight')
