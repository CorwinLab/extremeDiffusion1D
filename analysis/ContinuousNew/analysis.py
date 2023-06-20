import pandas as pd
import glob
from matplotlib import pyplot as plt
import json
import numpy as np
import sys 
import os
sys.path.append("../../dataAnalysis")
from continuousTheory import theoretical_mean, theoretical_variance, theoretical_long_time_variance

def calculateMeanVar(files):
    avg = None 
    squared = None
    num_of_files = 0
    maxTime = 10000
    for i, f in enumerate(files): 
        df = pd.read_csv(f)
        if max(df['Time']) < maxTime:
            print("Not enough time: ", f)
            continue 
        df = df[df['Time'] <= maxTime]
        time = df['Time'].values 
        if avg is None: 
            avg = df['Position'].values 
        else: 
            avg += df['Position'].values 
        num_of_files += 1

        if squared is None:
            squared = df['Position'].values ** 2 
        else: 
            squared += df['Position'].values ** 2 

    avg = avg / num_of_files
    var = squared / num_of_files - avg ** 2

    print("# of files:", num_of_files)
    return avg, var, time

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousNew/"
dirs = os.listdir(dir)
dirs.sort()
maxTime = 10000
'''
for d in dirs: 
    files = glob.glob(os.path.join(dir, d, "Max*.txt"))
    avg, var, time = calculateMeanVar(files)
    np.savetxt(os.path.join(dir, d, "Mean.txt"), avg)
    np.savetxt(os.path.join(dir, d, "Var.txt"), var)
    np.savetxt(os.path.join(dir, d, "Time.txt"), time)
'''

colors = ['tab:red', 'tab:blue', 'tab:green']
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r'$\mathrm{Mean}(\mathrm{Max}_t^N)$')

for i, d in enumerate(dirs): 
    vars_file = os.path.join(dir, d, "variables.json") 
    if d == '0.1':
        continue
    with open(vars_file) as f:
        vars = json.load(f)
    

    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    avg = np.loadtxt(os.path.join(dir, d, "Mean.txt"))

    N = vars['nParticles']
    D = vars['D']
    rc = vars['xi']
    sigma = vars['sigma']
    
    Dr = D + sigma / rc**2 / np.sqrt(2*np.pi)
    r0 = sigma / Dr
    
    tdagger = r0**2 * np.log(N)**2 / 4 / Dr
    tstar = r0**2 * np.log(N) / 4 / Dr
    ax.plot(time, avg, c=colors[i])
    #ax.plot(time, theoretical_mean(r0, D, N, time), ls='-.', label=r"$D$", c=colors[i])
    ax.plot(time, theoretical_mean(r0, Dr, N, time), ls='--', label=fr'$D_r = {Dr:10.2f}, D = {D}$', c=colors[i])
    #ax.vlines(tdagger, 1, 2 * 10**3, linestyle=':', label=r'$t^{\dagger}=$' + f'{tdagger:10.2f}', color=colors[i])
    #ax.vlines(tstar, 1, 2 * 10**3, linestyle='-.', label=r'$t^{*}=$' + f'{tstar:10.2f}', color=colors[i])
    
ax.set_ylim([1, 2*10**3])
ax.set_xlim([1, maxTime])
ax.legend()
ax.grid(True)
fig.savefig("Average.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1, maxTime])
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r'$\mathrm{Var}(\mathrm{Max}_t^N)$')

for i, d in enumerate(dirs): 
    vars_file = os.path.join(dir, d, "variables.json")
    if d == '0.1':
        continue
    with open(vars_file) as f:
        vars = json.load(f)
    
    N = vars['nParticles']
    D = vars['D']
    rc = vars['xi']
    sigma = vars['sigma']

    Dr = D + sigma / rc**2 / np.sqrt(2*np.pi)
    r0 = sigma / Dr
    tdagger = r0**2 * np.log(N)**2 / 4 / Dr
    tstar = r0**2 * np.log(N) / 4 / Dr
    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    var = np.loadtxt(os.path.join(dir, d, "Var.txt"))

    ax.plot(time, var, c=colors[i])
    #ax.plot(time, theoretical_variance(r0, D, N, time), ls='-.', label=r"$D$", c=colors[i])
    ax.plot(time, theoretical_variance(r0, Dr, N, time), ls='--', label=fr'$D_r = {Dr:10.2f}, D = {D}$', c=colors[i])
    #ax.plot(time, theoretical_long_time_variance(r0, Dr, N, time), ls='-.', label=r'$D_r$', c=colors[i])
    #ax.vlines(tdagger, 0.1, 2 * 10**4, linestyle=':', label=r'$t^{\dagger}=$' + f'{tdagger:10.2f}', color=colors[i])
    #ax.vlines(tstar, 0.1, 2 * 10**4, linestyle='-.', label=r'$t^{*}=$' + f'{tstar:10.2f}', color=colors[i])

ax.set_ylim([0.1, 2*10**4])
ax.legend()
ax.grid(True)
fig.savefig("Var.pdf", bbox_inches='tight')