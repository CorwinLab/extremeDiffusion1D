import sys
sys.path.append("../../src")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
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

directory = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassAbs"
dirs = os.listdir(directory)

run_again = False
if run_again:
    for d in dirs:
        path = os.path.join(directory, d, 'Q*.txt')
        files = glob.glob(path)
        df = calculate_mean(files, os.path.join(directory, d, "ConcatData.txt"), verbose=True)

        mean_file = os.path.join(directory, d, "Mean.txt")
        var_file = os.path.join(directory, d, "Var.txt")
        d, m, v = calculate_distance_mean(df, mean_file, var_file)

fontsize=12
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(dirs) / 1) for i in range(len(dirs))
]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\mathrm{Var}(\tau)$", fontsize=fontsize)
ax.set_xlabel(r"$x / \log(N)$", fontsize=fontsize)
print(dirs)
dirs = [int(x) for x in dirs]
dirs.sort()
for i, d in enumerate(dirs):
    print(d)
    N_exp = int(d)
    N = float(f"1e{N_exp}")
    logN = np.log(N).astype(float)
    path = os.path.join(directory, str(d))
    mean_file = os.path.join(path, "Mean.txt")
    var_file = os.path.join(path, "Var.txt")
    mean = np.loadtxt(mean_file).T
    var = np.loadtxt(var_file).T
    distance = mean[:, 0]
    assert (var[:, 0] == distance).all()
    var = var[:, 1]
    mean = mean[:, 1]
    ax.plot(distance / logN, var, label=N_exp, c=colors[i])

ax.grid(True)
ax.set_xlim([.5, 10**2])
xvals = np.linspace(.01, 2000, num=1000)
ax.legend()
fig.savefig("Variance.png", bbox_inches='tight')
