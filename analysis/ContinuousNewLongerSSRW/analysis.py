import pandas as pd
import glob
from matplotlib import pyplot as plt
import json
import numpy as np
import sys 
import os
from scipy.special import erf
sys.path.append("../../dataAnalysis")
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

dir = "/home/jacob/Desktop/talapasMount/JacobData/ContinuousNewLongerSSRW/"
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
N = 1000000
colors = ['r', 'b', 'g']
alpha=0.75

fig, ax = plt.subplots()
ax.set_xlabel(r"$t$")
ax.set_ylabel("Mean(Max)")
ax.set_xscale("log")
ax.set_yscale("log")

for i, d in enumerate(dirs): 
    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    mean = np.loadtxt(os.path.join(dir, d, "Mean.txt"))
    ax.plot(time, mean, label=f'D={d}', color=colors[i], alpha=alpha)
    #ax.plot(time, np.sqrt(4*float(d) * time * np.log(N)), ls='--', color='k')
    ax.plot(time, np.sqrt(4 * float(d) * np.log(N) * time), ls='-.', color=colors[i])

ax.set_xlim([1, maxTime])
ax.legend()
fig.savefig("Mean.pdf")

colors = ['r', 'b', 'g']
fig, ax = plt.subplots()
ax.set_xlabel(r"$t$")
ax.set_ylabel("Var(Max)")
ax.set_xscale("log")
ax.set_yscale("log")

for i, d in enumerate(dirs): 
    time = np.loadtxt(os.path.join(dir, d, "Time.txt"))
    var = np.loadtxt(os.path.join(dir, d, "Var.txt"))
    ax.plot(time, var, label=f'D={d}', color=colors[i], alpha=alpha)
    ax.plot(time, np.pi**2 / 6 * float(d) * time / np.log(N), ls='--', color=colors[i])

ax.set_xlim([1, maxTime])
ax.legend()
fig.savefig("Var.pdf")