import sys

sys.path.append("../../dataAnalysis")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import json
import pandas as pd

"""
Make figure plot showing environmental recovery
"""


db = Database()
einstein_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/EinsteinPaper/"
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
cdf_path = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/CDF/"
cdf_path_talapas = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/Paper/"
dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type="Max")
    N = int(path.split("/")[-1])
    # db.calculateMeanVar(path, verbose=True)

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")
# db.calculateMeanVar([cdf_path, cdf_path_talapas], verbose=True, maxTime=3453876)

db1 = db.getBetas(1)
for dir in db.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    #print(dir, " Systems:", x["number_of_systems"])

quantiles = db1.N(dir_type="Max")

fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{N})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))]
ypower = 0

for i, N in enumerate(quantiles):
    cdf_df, max_df = db1.getMeanVarN(N)
    max_df["Var Max"] = max_df["Var Max"] * 4
    max_df["Mean Max"] = max_df["Mean Max"] * 2
    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)

    var_theory = theory.quantileVar(
        Nquad, cdf_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
    )
    mean_theory = theory.quantileMean(Nquad, cdf_df["time"].values)
    decade_scaling = 25

    env_subtracted = max_df['Var Max'].values - theory.gumbel_var(max_df['time'].values, Nquad)
    env_time, env_recovered = theory.log_moving_average(max_df['time'].values, env_subtracted, window_size=10**(1/decade_scaling))

    #ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=colors[i])
    #ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)
    ax.plot(env_time[env_time > logN] / logN, env_recovered[env_time > logN], c=colors[i])
    
    dist_new, error = theory.log_moving_average(cdf_df['time'], cdf_df['Gumbel Mean Variance'] - theory.gumbel_var(cdf_df['time'].values, Nquad), 10**(1/decade_scaling))
    ax.plot(dist_new / logN, error, label=f'log_10(N)={N}', c=colors[i], alpha=0.5)

ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("EnvError.pdf", bbox_inches="tight")