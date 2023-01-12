import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os 
import sys 
import json
from matplotlib.colors import LinearSegmentedColormap
sys.path.append("../../dataAnalysis")
from theory import KPZ_mean_fit, KPZ_var_fit
from continuousTheory import theoretical_mean, theoretical_variance, theoretical_long_time_variance

tw_mean = -1.771 
tw_var = 0.813
gumbel_mean = 0.577
gumbel_var = np.pi**2 / 6

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/Continuous/1/"
folders = os.listdir(dir)
run_again = False
if run_again:
    for d in folders:
        files = glob.glob(os.path.join(dir, d, "Max*.txt"))
        max_dist = 10000
        mean = None
        var = None
        num_files = 0
        for f in files:
            print(f)
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

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(folders) / 1) for i in range(len(folders))]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Max}_t^N)$")
ax.set_xlim([1, 10000])

for i, d in enumerate(folders):
    if not os.path.exists(dir + d + "/Mean.txt"):
        print('No Mean file found for:', d)
        continue
    mean = np.loadtxt(dir + d + "/Mean.txt")
    time = np.loadtxt(dir + d + "/Time.txt")
    with open(os.path.join(dir, d, 'variables.json')) as f:
        vars = json.load(f)
    N = vars['nParticles'] 
    D = vars['D'] 
    r0 = vars['xi']
    print(vars['Date'])
    ax.plot(time, mean, label=fr"$r_c={r0}, D={D}$", c=colors[i])
    D = 2 * D 
    r0 = np.sqrt(2 * np.pi) / 2 * r0

    ax.plot(time, theoretical_mean(r0, D, N, time), ls='--', c=colors[i])

    xvals = np.array([100, 10000])
    ax.plot(xvals, np.sqrt(4 * D/2 * np.log(N) * xvals), c='k', ls='--', label=r'$\sqrt{4 D \log(N) t}$')

ax.grid(True)
ax.legend()
fig.savefig("DSweepMean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}_t^N)$")
ax.set_title(r"$N=10,000$")
ax.set_xlim([1, 100000])
for i, d in enumerate(folders):
    if not os.path.exists(dir + d + "/Mean.txt"):
        print('No Mean file found for:', d)
        continue
    var = np.loadtxt(dir + d + "/Var.txt")
    time = np.loadtxt(dir + d + "/Time.txt")
    with open(os.path.join(dir, d, 'variables.json')) as f:
        vars = json.load(f)
    N = vars['nParticles'] 
    D = vars['D'] 
    r0 = vars['xi']
    ax.plot(time, var, label=fr"$r_c={r0}, D={D}$", c=colors[i])

    D = 2 * D 
    r0 = np.sqrt(2 * np.pi) / 2 * r0
    ax.plot(time, theoretical_variance(r0, D, N, time), ls='--', c=colors[i])
    ax.plot(time, theoretical_long_time_variance(r0, D, N, time), ls='-.', c=colors[i])

ax.set_xlim([1, 10000])

ax.grid(True)
ax.legend()
fig.savefig("DSweepVar.pdf", bbox_inches='tight')
