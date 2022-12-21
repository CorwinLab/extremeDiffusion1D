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
Make figure plot showing sampling error
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
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\mathrm{Sam}_t^{N}) - \mathrm{Var}^{\mathrm{Asy}}(\mathrm{Sam}_t^{N})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))]
ypower = 0

for i, N in enumerate(quantiles):
    cdf_df, _ = db1.getMeanVarN(N)

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)

    decade_scaling = 25
    dist_new, error = theory.log_moving_average(cdf_df['time'], cdf_df['Gumbel Mean Variance'] - theory.gumbel_var(cdf_df['time'].values, Nquad), 10**(1/decade_scaling))
    ax.plot(dist_new / logN, error, label=f'log_10(N)={N}', c=colors[i], alpha=0.5)

xvals = np.array([10**2, 500 * 10**3])
ax.plot(xvals, xvals / 10, ls='--', c='k', label=r'$\propto t$')
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
ax.legend()
ax.grid(True)
fig.savefig("SamplingError.pdf", bbox_inches="tight")