import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os 

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousCorrected/"
folders = os.listdir(dir)
for d in folders:
    files = glob.glob(f"/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousCorrected/{d}/Max*.txt")
    max_dist = 10000
    mean = None
    var = None
    num_files = 0
    for f in files:
        try:
            data = np.loadtxt(f, skiprows=1, delimiter=',')
        except:
            continue

        if data[-1, 0] < max_dist:
            continue
        pos = data[:, 0]
        num_files += 1
        if mean is None: 
            mean = data[:, 1]
        else: 
            mean += data[:, 1]

        if var is None: 
            var = data[:, 1]**2
        else: 
            var += data[:, 1]**2

    mean /= num_files
    var = var / num_files - mean**2
    np.savetxt(dir + d + "/Mean.txt", mean)
    np.savetxt(dir + d + "/Var.txt", var)
    np.savetxt(dir + d + "/Position.txt", pos)

xvals = np.array([100, 1000])

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Position")
ax.set_ylabel("Mean Maximum")
ax.set_xlim([1, 10000])

for d in folders: 
    mean = np.loadtxt(dir + d + "/Mean.txt")
    pos = np.loadtxt(dir + d + "/Position.txt")
    ax.plot(pos, mean, label=fr"$\xi={d}$")

#ax.plot(xvals, np.sqrt(xvals) * 10, c='k', ls='--', label=r'$\sqrt{x}$')
ax.grid(True)
ax.legend()
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Position")
ax.set_ylabel("Variance of Maximum")
ax.set_title(r"$D=1, N=10,000$")
ax.set_xlim([1, 10000])
for d in folders: 
    var = np.loadtxt(dir + d + "/Var.txt")
    pos = np.loadtxt(dir + d + "/Position.txt")
    ax.plot(pos, var, label=fr"$\xi={d}$")

#ax.plot(xvals, np.sqrt(xvals)*50, c='k', ls='--', label=r'$\sqrt{x}$')
#ax.plot(xvals, xvals*10, ls='--', c='r', label=r'$x$')
ax.grid(True)
ax.legend()
fig.savefig("Var.png", bbox_inches='tight')