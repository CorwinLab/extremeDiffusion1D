import sys
sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import json
import glob
import os
import pandas as pd

def calculate_mean(files, save_file, verbose=True):
    df_tot = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f, delimiter=',')
        df_tot = pd.concat([df_tot, df])
        if verbose:
            print(f)

    df_tot.reset_index(inplace=True, drop=True)
    df_tot.to_csv(save_file)
    return df_tot

def calculate_distance_mean(df, mean_save, var_save):
    unique_distances = np.unique(df['Distance'].values)

    mean = []
    var = []

    for d in unique_distances:
        df_d = df[df['Distance'] == d]
        mean.append(np.mean(df_d['Time'].values))
        var.append(np.var(df_d['Time'].values))

    np.savetxt(mean_save, np.array([unique_distances, mean]))
    np.savetxt(var_save, np.array([unique_distances, var]))
    return unique_distances, mean, var

directory = "/home/jacob/Desktop/talapasMount/JacobData/FirstPass/Q*.txt"
files = glob.glob(directory)

run_again = False
if run_again:
    dfe = calculate_mean(files, 'CompiledData.txt', verbose=True)
else:
    dfe = pd.read_csv("CompiledData.txt")

directory = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassUniform/Q*.txt"
files = glob.glob(directory)

if run_again:
    dfu = calculate_mean(files, 'UniformCompiledData.txt', verbose=True)
else:
    dfu = pd.read_csv("UniformCompiledData.txt")


distance_u, mean_u, var_u = calculate_distance_mean(dfu, 'UMean.txt', 'UVar.txt')
distance_e, mean_e, var_e = calculate_distance_mean(dfe, 'EMean.txt', 'EVar.txt')

N_exp = 24
N = np.quad(f"1e{N_exp}")
logN = np.log(N).astype(float) / np.log(2)

RWRESam = np.loadtxt("../FirstPassageCDF/AveragedData.txt")
theoretical_distances = np.loadtxt("distances.txt")
theoretical_variance = np.loadtxt("variance.txt")
theoretical_variance = (theoretical_variance / 2)**2

fig, ax = plt.subplots()
ax.plot(distance_u / logN, mean_u)
ax.plot(distance_e / logN, mean_e)
ax.set_xlabel("Distance / log2(N)")
ax.set_ylabel("Mean(First Passage Time)")
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Mean.png")

fig, ax = plt.subplots()
ax.plot(distance_u / logN, var_u, label='Max RWRE')
ax.plot(distance_e / logN, var_e, label='Max SSRW')
ax.plot(RWRESam[:, 0] / logN, RWRESam[:, 2], label='Sampling RWRE')
ax.plot(distance_e[-1000:] / logN, (distance_e[-1000:])**4 / 10**8, c='k', label=r'$t^{4}$')
ax.plot(theoretical_distances / logN, theoretical_variance, c='m')
ax.set_xlabel("Distance / log2(N)")
ax.set_ylabel("Var(First Passage Time)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("Var.png")
