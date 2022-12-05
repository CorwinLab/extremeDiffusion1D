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
nFiles = [1000, 1000, 1000, 501, 501]

for i, N in enumerate(quantiles):
    cdf_df, _ = db1.getMeanVarN(N)
    print("Reading: ", os.path.join(directory, str(N), f'MeanVar{nFiles[i]}.txt'))
    max_df = pd.read_csv(os.path.join(directory, str(N), f'MeanVar{nFiles[i]}.txt'))
    
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

    ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=colors[i])
    ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)
    ax.plot(env_time[env_time > logN] / logN, env_recovered[env_time > logN], c=colors[i])
    #ax.scatter(x=logN**2 / logN, y=theory.quantileVar(Nquad, logN**2, crossover=logN**(1.5), width=logN**(4/3)), color=colors[i], edgecolor='k', marker='*', s=50, linewidth=0.005, zorder=99)

ax.plot([100, 1000], (np.array([100, 1000]))**(1/2) * 90, c='k', alpha=0.6)
ax.annotate(r"$\propto t^{\frac{1}{2}}$", xy=(100, 5000), fontsize=fontsize+3)

ax.plot([2, 30], 100 * np.array([2, 30])**(1/3), c='k', alpha=0.6)
ax.annotate(r"$\propto t^{\frac{1}{3}}$", xy=(20, 5000), fontsize=fontsize+3)

#x, y
x_shift = 12
y_shift = 3.5
start_coord = (250 / x_shift, 6 / y_shift)
end_coord = (75 / x_shift, 3 * 10 ** 3 / y_shift)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx)) + 5
ax.annotate(
    "",
    xy=end_coord,
    xytext=start_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=50,
        headwidth=85,
        headlength=40,
        alpha=0.3,
    ),
    zorder=0,
)
ax.annotate(
    r"$N=10^{2}$",
    xy=(155 / x_shift, 2.6 / y_shift),
    c=colors[0],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)
ax.annotate(
    r"$N=10^{300}$",
    xy=(40 / x_shift, 2.5 * 10 ** 3 / y_shift),
    c=colors[-1],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)

'''
axins = ax.inset_axes([0.615, 0.05, 0.35, 0.35])
axins.plot(cdf_df["time"] / logN, var_theory / logN ** (ypower), "--", c=colors[i])
axins.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"] / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
)
axins.plot(env_time / logN, env_recovered, c=colors[i])
axins.set_xlim([2 * 10 ** 2, 3 * 10 ** 2])
axins.set_ylim([3.2 * 10 ** 2, 5.1 * 10 ** 2])
axins.set_xscale("log")
axins.set_yscale("log")
axins.xaxis.set_ticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
'''
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("QuantileVar.pdf", bbox_inches="tight")