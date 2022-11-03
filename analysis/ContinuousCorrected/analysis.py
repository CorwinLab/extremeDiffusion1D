import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os 

def theoretical_mean(r0, D, N, t):
    return 4 * np.sqrt(6) * D / r0 * np.sqrt(np.sqrt(1+ r0**2 *np.log(N) / 12 / D / t) - 1) * t

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousLonger/"
folders = os.listdir(dir)
for d in folders:
    files = glob.glob(f"/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousLonger/{d}/Max*.txt")
    max_dist = 100000
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

        time = data[:, 0]
        num_files += 1
        if mean is None: 
            mean = data[:, 1]
        else: 
            mean += data[:, 1]

        if var is None: 
            var = data[:, 1]**2
        else: 
            var += data[:, 1]**2
    if num_files == 0:
        continue
    mean /= num_files
    var = var / num_files - mean**2
    np.savetxt(dir + d + "/Mean.txt", mean)
    np.savetxt(dir + d + "/Var.txt", var)
    np.savetxt(dir + d + "/Time.txt", time)

xvals = np.array([100, 1000])
N = 100000
D = 2

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Maximum")
ax.set_xlim([1, 100000])

for r0 in folders:
    if not os.path.exists(dir + r0 + "/Mean.txt"):
        continue
    mean = np.loadtxt(dir + r0 + "/Mean.txt")
    time = np.loadtxt(dir + r0 + "/Time.txt")
    ax.plot(time, mean, label=fr"$r_c={r0}$")
    r0 = float(r0)
    ax.plot(time, theoretical_mean(2 * np.sqrt(np.pi) * r0, D, N, time), ls='--')
    #ax.plot(np.geomspace(1, 10**5), np.sqrt(4 * D * np.log(N) * np.geomspace(1, 10**5)))

#ax.plot(xvals, np.sqrt(xvals) * 10, c='k', ls='--', label=r'$\sqrt{x}$')
ax.grid(True, 'both')
ax.legend()
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Variance of Maximum")
ax.set_title(r"$D=1, N=10,000$")
ax.set_xlim([1, 100000])
for r0 in folders:
    if not os.path.exists(dir + r0 + "/Mean.txt"):
        continue
    var = np.loadtxt(dir + r0 + "/Var.txt")
    time = np.loadtxt(dir + r0 + "/Time.txt")
    ax.plot(time, var, label=fr"$r_c={r0}$")

#ax.plot(xvals, np.sqrt(xvals)*50, c='k', ls='--', label=r'$\sqrt{x}$')
#ax.plot(xvals, xvals*10, ls='--', c='r', label=r'$x$')
ax.grid(True)
ax.legend()
fig.savefig("Var.png", bbox_inches='tight')