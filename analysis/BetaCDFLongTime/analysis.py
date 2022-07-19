import numpy as np
import npquad 
from matplotlib import pyplot as plt
import glob
import os
import sys
sys.path.append("../../src")
import theory
from overalldatabase import Database

varTW = 0.813

def calculateMeanVar(files):
    mean = np.zeros(shape=(3299, 11))
    squared = np.zeros(shape=(3299, 11))
    count = 0
    for f in files: 
        print(f)
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        if data[-1, 0] != 22701:
            continue
        mean+=data
        squared += data**2

        count+=1
    mean = mean / count
    var = squared / count - mean**2
    time = mean[:, 0] + 1
    return time, mean, var

def varPowerLaw(beta, t, N):
    logN = np.log(N).astype(float)
    return varTW * (t*logN/2)**(1/3) * beta**(-4/3)

dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaCDFLongTime/"
dirs = os.listdir(dir)
run_again = False

'''Make plot of env'''
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel("Mean(Env)")

figv, axv = plt.subplots()
axv.set_xscale("log")
axv.set_yscale("log")
axv.set_xlabel("t / log(N)")
axv.set_ylabel("Var(Env)")
logN = np.log(1e24)
N = 1e24

colors = ['r', 'b', 'c', 'm']

for i, beta in enumerate([0.01, 0.1, 1, 10]): 
    if run_again: 
        path = os.path.join(dir, str(beta), 'Q*.txt')
        files = glob.glob(path)
        time, mean, var = calculateMeanVar(files)
        np.savetxt(f"Data/Mean{beta}.txt", mean)
        np.savetxt(f"Data/Var{beta}.txt", var)
        np.savetxt(f"Data/Time{beta}.txt", time)
    mean_file = f"Data/Mean{beta}.txt"
    var_file = f"Data/Var{beta}.txt"
    time_file = f"Data/Time{beta}.txt"

    mean = np.loadtxt(mean_file)
    var = np.loadtxt(var_file)
    time = np.loadtxt(time_file)

    ax.plot(time / logN, mean[:,3], label=beta)
    axv.plot(time / logN, var[:,3], label=beta, c=colors[i], alpha=1)
    #axv.plot(time / logN, varPowerLaw(beta, time, N), c=colors[i], ls='--')
    #axv.plot(time / logN, theory.quantileVarLongTimeBetaDist(N, time, beta), ls='-.', c=colors[i])

ax.set_xlim([min(time/logN), max(time/logN)])
axv.set_xlim([min(time/logN), max(time/logN)])

ax.legend()
axv.legend()
fig.savefig("Mean.png")
figv.savefig("Var.png")

'''Make plot of sampling'''
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / log(N)")
ax.set_ylabel("Var(Sam)")

for beta in [0.01, 0.1, 1, 10]: 
    mean_file = f"Data/Mean{beta}.txt"
    var_file = f"Data/Var{beta}.txt"
    time_file = f"Data/Time{beta}.txt"

    mean = np.loadtxt(mean_file)
    var = np.loadtxt(var_file)
    time = np.loadtxt(time_file)

    ax.plot(time / logN, mean[:,8], label=beta)

ax.set_xlim([min(time/logN), max(time/logN)])
ax.set_ylim([10**-2, 1000])
ax.legend()
fig.savefig("SamplingVar.png")

db = Database()
beta_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    if beta == 0.01:
        continue
    db.add_directory(path, dir_type='Max')



betas = db.betas()
N_exp = db.N(dir_type='Max')[0] # Should be all the same beta
N = np.quad(f"1e{N_exp}")
logN = np.log(N)

betas.sort()
for i, beta in enumerate(betas):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t/log(N)")
    ax.set_ylabel("Var")

    if beta not in [0.01, 0.1, 1, 10]:
        continue 
    dbb = db.getBetas(beta)
    _, max_df = dbb.getMeanVarN(N_exp)
    max_df['Mean Max'] *= 2
    max_df['Var Max'] *= 4

    ax.plot(max_df['time'] / logN, max_df['Var Max'], label='Max')
    
    if beta==1 or beta==10: 
        beta = int(beta)
    mean_file = f"Data/Mean{beta}.txt"
    var_file = f"Data/Var{beta}.txt"
    time_file = f"Data/Time{beta}.txt"

    mean = np.loadtxt(mean_file)
    var = np.loadtxt(var_file)
    time = np.loadtxt(time_file)

    ax.plot(time / logN, var[:, 3], label='Env')
    ax.plot(time / logN, mean[:, 8], label='Sam')
    ax.plot(time / logN, var[:, 3] + mean[:, 8], label='Env + Sam')
    ax.set_ylim([10**-1, 10**5])
    ax.legend()
    ax.grid(True)
    fig.savefig(f"Total{beta}.png", bbox_inches='tight')


def quantile(N, time): 
    logN = np.log(N).astype(float)
    return (logN / time)**(2/3) * 2**(2/3) * (1- logN/time)**(4/3) / (1- (1-logN/time)**2)

time = np.logspace(1, 6, num=5000)
N = 1e24
logN = np.log(N).astype(float)
fig, ax = plt.subplots()
ax.plot(time / logN, theory.quantileVarShortTime(N, time))
ax.plot(time / logN, quantile(N, time))
ax.plot(time / logN, (time/logN/2)**(1/3))
ax.plot(time / logN, theory.quantileVar(N, time))
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True)
fig.savefig("QuantileVar.png")