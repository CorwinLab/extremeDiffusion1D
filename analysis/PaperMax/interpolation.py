import sys

sys.path.append("../../dataAnalysis")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import json

db = Database()
''' I moved these to /corwinLabMount/MaxParticlePaperData/ '''

einstein_dir = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/JacobData/EinsteinPaper/"
directory = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/Paper/Max/"
cdf_path = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/Paper/CDF/"
cdf_path_talapas = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/JacobData/Paper/"

dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type="Max")

e_dirs = os.listdir(einstein_dir)
for dir in e_dirs:
    path = os.path.join(einstein_dir, dir)
    N = int(path.split("/")[-1])
    if N == 300:
        continue
    db.add_directory(path, dir_type="Max")

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")

db1 = db.getBetas(1)
for dir in db1.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    print(dir, " Systems:", x["number_of_systems"])

dbe = db.getBetas(np.inf)

"""
Long time plot
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{N})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
i = 3
cdf_df, _ = db1.getMeanVarN(N)

cdf_df['time'] += 2
print(cdf_df)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df["time"].values
var_theory = np.piecewise(
    time,
    [time < logN, time >= logN],
    [lambda t: 0, lambda t: theory.quantileVarShortTime(Nquad, t)],
)
var_long = np.piecewise(
    time,
    [time < logN, time >= logN],
    [lambda t: np.nan, lambda t: theory.quantileVarLongTime(Nquad, t)],
)

center = logN ** (1 / 2)
width = 1.3 * logN ** (4 / 3) / logN

# ax2 = fig.add_axes([0.53, 0.13, 0.35, 0.35])
# ax2.set_xscale("log")
# ax2.set_yscale("log")
# ax2.set_xlim([center - width, center + width])
# ax2.set_ylim([3 * 10 ** 1, 9 * 10 ** 1])

ax.plot(cdf_df["time"] / logN, var_theory, "--", c="tab:orange")
ax.plot(cdf_df["time"] / logN, var_long, "--", c="darkviolet")
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"],
    label=N,
    c='tab:blue',
    alpha=0.75,
    zorder=0,
)
xs = np.array([3 * 10 ** 2, 7 * 10 ** 3])
ys = 13 * xs ** (1 / 3)
ys2 = 19 * xs ** (1 / 2)
ax.plot(xs, ys, c="tab:orange")
ax.plot(xs, ys2, c="darkviolet")
# ax.arrow(x=logN**2 / logN, y=10**-1, dx=0, dy=0.1, color='k', width=20, head_length=0.3)
ax.vlines(x=logN ** 2 / logN, ymin=10 ** -1, ymax=10 ** 4, color="k", ls=":", alpha=0.4)
ax.annotate(r"$t=(\log (N))^2$", xy=(10 ** 2, 10 ** 3), c="k", fontsize=fontsize)
ax.annotate(
    r"$\propto {t}^{\frac{1}{2}}$",
    xy=(2 * 10 ** 3, 2 * 10 ** 3),
    c="k",
    fontsize=fontsize + 3,
)
ax.annotate(
    r"$\propto {t}^{\frac{1}{3}}$", xy=(2 * 10 ** 3, 40), c="k", fontsize=fontsize + 3
)

# ax2.plot(cdf_df["time"] / logN, var_theory, "--", c="r")
# ax2.plot(cdf_df["time"] / logN, var_long, "--", c="b")
# ax2.plot(
#     cdf_df["time"] / logN,
#     cdf_df["Var Quantile"],
#     label=N,
#     c='b',
#     alpha=0.5,
#     zorder=0,
# )

ax.set_xlim([0.3, 10 ** 4])
ax.set_ylim([10 ** -1, 10 ** 4])
#ax.indicate_inset_zoom(ax2, edgecolor="black")

fig.savefig("Interpolation.svg", bbox_inches="tight")
