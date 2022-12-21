import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os 
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_mean_fit, KPZ_var_fit

tw_mean = -1.771 
tw_var = 0.813
gumbel_mean = 0.577
gumbel_var = np.pi**2 / 6

def theoretical_mean(r0, D, N, t):
    gamma = np.log(N) / t
    return np.sqrt(4*D*np.log(N) * t) + 0.5 * np.sqrt(D * t / np.log(N))

def theoretical_var(r0, D, N, t): 
    gamma = np.log(N) / t
    return (1/2 * r0**(2/3) * (4*D*gamma)**(1/6)*t**(1/3))**2 * tw_var + D / gamma * np.pi**2 / 6

def theoretical_mean_long_time(r0, D, N, t):
    g = r0 * np.log(N) / np.sqrt(4 * D * t)
    return r0 * np.sqrt(g) * (4 * D * t / r0**2)**(3/4) + r0 / 2 / np.sqrt(g) * (4*D*t/r0**2)**(1/4) * (gumbel_mean + KPZ_mean_fit(g**2))

def theoretical_var_long_time(r0, D, N, t):
    g = r0 * np.log(N) / np.sqrt(4 * D * t)
    return (r0 / 2 / np.sqrt(g) * (4*D*t/r0**2)**(1/4))**2 * (gumbel_var + KPZ_var_fit(g**2))

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/Continuous/"
folders = os.listdir(dir)
for d in folders:
    files = glob.glob(f"/home/jacob/Desktop/corwinLabMount/CleanData/Continuous/{d}/Max*.txt")
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

        time = data[:, 0]
        if mean is None: 
            mean = data[:, 1]
        else: 
            mean += data[:, 1]

        if var is None: 
            var = data[:, 1]**2
        else: 
            var += data[:, 1]**2
        num_files += 1
    
    if num_files == 0:
        continue
    print("Number of files:", num_files)
    mean /= num_files
    var = var / num_files - mean**2
    np.savetxt(dir + d + "/Mean.txt", mean)
    np.savetxt(dir + d + "/Var.txt", var)
    np.savetxt(dir + d + "/Time.txt", time)

xvals = np.array([100, 1000])
N = 100000
D = 1

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Maximum")
ax.set_xlim([1, 10000])

for r0 in folders:
    if not os.path.exists(dir + r0 + "/Mean.txt"):
        continue
    mean = np.loadtxt(dir + r0 + "/Mean.txt")
    time = np.loadtxt(dir + r0 + "/Time.txt")
    ax.plot(time, mean, label=fr"$r_c={r0}, D=1$")
    r0 = float(r0)
    ax.plot(time, theoretical_mean(2 * np.sqrt(np.pi) * r0, D, N, time), ls='--')
    ax.plot(time, theoretical_mean_long_time(2 * np.sqrt(np.pi) * r0, D, N, time), ls='-.')

#ax.plot(xvals, np.sqrt(xvals) * 10, c='k', ls='--', label=r'$\sqrt{x}$')
ax.grid(True)
ax.legend()
fig.savefig("Mean.pdf", bbox_inches='tight')

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
    r0 = float(r0)
    ax.plot(time, theoretical_var(2 * np.sqrt(np.pi) * r0, D, N, time), ls='--')
    ax.plot(time, theoretical_var_long_time(2 * np.sqrt(np.pi) * r0, D, N, time), ls='-.')

ax.set_xlim([1, 10000])

ax.grid(True)
ax.legend()
fig.savefig("Var.pdf", bbox_inches='tight')