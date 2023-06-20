import pandas as pd
import glob
from matplotlib import pyplot as plt
import json
import numpy as np
import sys 
import os
sys.path.append("../../dataAnalysis")
from numericalMaximum import getNParticleMeanVar
from continuousTheory import theoretical_mean, theoretical_variance, theoretical_long_time_variance

def calculateMeanVar(files, maxTime):
    avg = None 
    squared = None
    num_of_files = 0
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
        print(f)
    avg = avg / num_of_files
    var = squared / num_of_files - avg ** 2

    print("# of files:", num_of_files)
    return avg, var, time

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousNewLonger/"
dirs = os.listdir(dir)
dirs.sort()
maxTime = 100_000
'''
for d in dirs: 
    files = glob.glob(os.path.join(dir, d, "Max*.txt"))
    avg, var, time = calculateMeanVar(files, maxTime)
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
    with open(vars_file) as f:
        vars = json.load(f)

    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    avg = np.loadtxt(os.path.join(dir, d, "Mean.txt"))
    ax.plot(time, avg, c=colors[i])

    N = vars['nParticles']
    D = vars['D']
    rc = vars['xi']
    sigma = vars['sigma']
    Dr = D + sigma / rc**2 / np.sqrt(2*np.pi) / 2
    r0 = sigma / Dr

    #mean, var = getNParticleMeanVar(time, N, Dr, 'Classical', 1)
    ax.plot(time, theoretical_mean(r0, Dr, N, time), ls='--', label=fr'$D_r = {Dr:10.2f}, D = {D}$', c=colors[i])

ax.set_title(r"$r_c = 1, \sigma=1$")
ax.set_ylim([1, 10**4])
ax.set_xlim([1, maxTime])
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
    with open(vars_file) as f:
        vars = json.load(f)
    
    N = vars['nParticles']
    D = vars['D']
    rc = vars['xi']
    sigma = vars['sigma']

    Dr = D + sigma / rc**2 / np.sqrt(2*np.pi) / 2
    r0 = sigma / Dr
    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    var = np.loadtxt(os.path.join(dir, d, "Var.txt"))

    ax.plot(time, var, c=colors[i])

    ax.plot(time, theoretical_variance(r0, Dr, N, time), ls='--', label=fr'$D_r = {Dr:10.2f}, D = {D}$', c=colors[i])


ax.set_title(r"$r_c = 1, \sigma=1$")
ax.set_ylim([0.1, 2*10**5])
ax.grid(True)
fig.savefig("Var.pdf", bbox_inches='tight')