import sys
sys.path.append("../../src")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import glob
import os
import pandas as pd
from scipy.stats import linregress

def calculate_mean(files, save_file, verbose=True):
    df_tot = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f, delimiter=',')
        if max(df['Distance']) != 5526:
            continue

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


if __name__ == '__main__':
    dir = "/home/jacob/Desktop/talapasMount/JacobData/QuenchedFirstPassageDiscrete"

    run_again = False
    if run_again:
        path = os.path.join(dir, 'Q*.txt')
        files = glob.glob(path)
        df = calculate_mean(files, os.path.join(dir, "ConcatData.txt"), verbose=True)
        
        mean_file = os.path.join(dir, "Mean.txt")
        var_file = os.path.join(dir, "Var.txt")
        d, m, v = calculate_distance_mean(df, mean_file, var_file)

    N_exp = 24
    N = float(f"1e{N_exp}")
    logN = np.log(N).astype(float)
    mean_file = os.path.join(dir, "Mean.txt")
    var_file = os.path.join(dir, "Var.txt")
    mean = np.loadtxt(mean_file).T
    var = np.loadtxt(var_file).T
    distance = mean[:, 0]
    assert (var[:, 0] == distance).all()
    var = var[:, 1]
    mean = mean[:, 1]

    dir = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassDiscreteAbs"
    timerandomMean = np.loadtxt(os.path.join(dir, "Mean.txt")).T
    timerandomDistance = timerandomMean[:, 0]
    timerandomMean = timerandomMean[:, 1]
    timerandomVar = np.loadtxt(os.path.join(dir, "Var.txt")).T
    timerandomVar = timerandomVar[:, 1]

    fontsize=12
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    alpha=1

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathrm{Var}(\tau)$", fontsize=fontsize)
    ax.set_xlabel(r"$x / \log(N)$", fontsize=fontsize)
    ax.plot(distance / logN, var, label=r'$\mathrm{Var}(\tau_{quenched})$', alpha=alpha, c='r')
    ax.plot(timerandomDistance / logN, timerandomVar, label=r'$\mathrm{Var}(\tau_{RWRE})$', alpha=alpha, c='k')
    ax.legend()
    fig.savefig("Var.pdf", bbox_inches='tight')
    
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathrm{Mean}(\tau)$", fontsize=fontsize)
    ax.set_xlabel(r"$x / \log(N)$", fontsize=fontsize)
    ax.plot(distance / logN, mean, label=r'$\mathrm{Var}(\tau_{quenched})$', alpha=alpha, c='r')
    ax.plot(timerandomDistance / logN, timerandomMean, label=r'$\mathrm{Var}(\tau_{RWRE})$', alpha=alpha, c='k')

    ax.legend()
    fig.savefig("Mean.pdf", bbox_inches='tight')

