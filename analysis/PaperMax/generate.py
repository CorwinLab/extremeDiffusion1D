import sys

sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import json
import os

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

e_dirs = os.listdir(einstein_dir)
for dir in e_dirs:
    path = os.path.join(einstein_dir, dir)
    N = int(path.split("/")[-1])
    if N == 300:
        continue
    db.add_directory(path, dir_type="Max")
    # db.calculateMeanVar(path, verbose=True)

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")
# db.calculateMeanVar([cdf_path, cdf_path_talapas], verbose=True, maxTime=3453876, nFiles=500)

db1 = db.getBetas(1)
for dir in db.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    print(dir, " Systems:", x["number_of_systems"])

dbe = db.getBetas(float("inf"))
quantiles = db1.N(dir_type="Max")

"""
Make the maximum variance and mean plots
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}_t^{N})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

ax2 = fig.add_axes([0.53, 0.21, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax2.set_ylabel(r"$\mathrm{Mean}(\mathrm{Max}_t^{N})$", labelpad=0, fontsize=fontsize)
ax2.tick_params(axis="both", which="major", labelsize=fontsize)
ax2.set_xlim([10 ** -3, 5 * 10 ** 3])
ax2.set_ylim([1, 10 ** 5])

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))]
ypower = 0
for i, N in enumerate(quantiles):
    cdf_df, max_df = db1.getMeanVarN(N)

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)
    max_df["Var Max"] = max_df["Var Max"] * 4
    max_df["Mean Max"] = max_df["Mean Max"] * 2

    var_theory = theory.quantileVar(
        Nquad, max_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
    )
    mean_theory = theory.quantileMean(Nquad, max_df["time"].values)

    ax.plot(
        max_df["time"] / logN,
        (var_theory + theory.gumbel_var(max_df["time"].values, Nquad))
        / logN ** (ypower),
        "--",
        c=colors[i],
    )
    ax.plot(
        max_df["time"] / logN,
        max_df["Var Max"] / logN ** (ypower),
        label=N,
        c=colors[i],
        alpha=0.5,
    )

    ax2.plot(max_df["time"] / logN, max_df["Mean Max"], c=colors[i], alpha=0.8)
    ax2.plot(max_df["time"] / logN, mean_theory, "--", c=colors[i])

# x, y
start_coord = (20, 6)
end_coord = (6, 9 * 10 ** 2)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])

theta = np.rad2deg(np.arctan2(dy, dx))+5
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=80, headlength=40, alpha=0.3), zorder=0)
ax.annotate(r"$N=10^{2}$", xy=(12.5, 2.6), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor', fontsize=fontsize)
ax.annotate(r"$N=10^{300}$", xy=(3, 8*10**2), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor', fontsize=fontsize)

# Make a linear plot
ax.plot([100, 1000], np.array([100, 1000]) / 10, c='k', alpha=0.6)
ax.annotate(r"$\propto t$", xy=(2*10**3, 40), c='k', fontsize=fontsize+3)

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-1, 10**4])
ax2.remove()
fig.savefig("MaxVar.pdf", bbox_inches="tight")

"""
Make plot showing the recovery of environmental data
"""
fontsize = 12
alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

decade_scaling = 25
env_subtracted = max_df["Var Max"].values - theory.gumbel_var(
    max_df["time"].values, Nquad
)
env_time, env_recovered = theory.log_moving_average(
    max_df["time"].values, env_subtracted, N, window_size=10 ** (1 / decade_scaling)
)
max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
einsten_color = "tab:purple"

var_theory = theory.quantileVar(
    Nquad, cdf_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
)
max_var_theory = theory.quantileVar(
    Nquad, max_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
) + theory.gumbel_var(max_df["time"].values, Nquad)
ax.plot(
    max_df["time"] / logN,
    max_df["Var Max"],
    c=max_color,
    alpha=alpha,
    label=r"$\mathrm{Var}(\mathrm{Max}_t^N)$",
)
ax.plot(max_df["time"] / logN, max_var_theory, c=max_color, ls="--")
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"],
    c=quantile_color,
    alpha=alpha,
    label=r"$\mathrm{Var}(\mathrm{Env}_{t}^N)$",
)
ax.plot(cdf_df["time"] / logN, var_theory, ls="--", c=quantile_color)
ax.plot(
    cdf_df["time"] / logN,
    theory.gumbel_var(cdf_df["time"].values, Nquad),
    c=gumbel_color,
    ls="--",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Gumbel Mean Variance"],
    c=gumbel_color,
    alpha=alpha,
    label=r"$\mathrm{Var}(\mathrm{Sam}_t^N)$",
)
ax.plot(
    env_time[env_time > logN] / logN,
    env_recovered[env_time > logN],
    c="tab:orange",
    alpha=alpha,
    label=r"$\mathrm{Var}(\mathrm{Max}_t^N) - \mathrm{Var}(\mathrm{Sam}_t^N)$",
)

cdf_df, max_df = dbe.getMeanVarN(N)
max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2
np.savetxt(
    "7_time.txt",
    max_df["time"][max_df["time"] > np.log2(float(f"1e{N}")).astype(float)].values,
)
ax.plot(
    max_df["time"] / logN,
    max_df["Var Max"],
    alpha=alpha,
    c=einsten_color,
    label=r"SSRW $\mathrm{Var}(\mathrm{Max}_t^N)$",
)
c1 = np.loadtxt("7_c1.dat")
einstein_theory = theory.einstein_var(Nquad, c1)
ax.plot(
    max_df["time"][max_df["time"] > np.log2(float(f"1e{N}")).astype(float)] / logN,
    einstein_theory,
    c=einsten_color,
    ls="--",
    zorder=0,
)

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, quantile_color, gumbel_color, "tab:orange", einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)

ax.vlines(x=logN ** 2 / logN, ymin=10 ** -1, ymax=10 ** 4, color="k", ls=":", alpha=0.4)
ax.annotate(
    r"$t=(\log(N))^2$", xy=(logN ** 2 / logN, 10 ** 3), color="k", fontsize=fontsize
)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("MaxQuantComp.pdf", bbox_inches="tight")

"""
Make plot showing max mean
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Max}_t^{N})$", fontsize=fontsize, labelpad=0)
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

    mean_theory = theory.quantileMean(Nquad, max_df["time"].values)
    mean_theory_long = theory.quantileMeanLongTime(Nquad, max_df["time"].values)

    ax.plot(max_df["time"] / logN, mean_theory / logN ** (ypower), "--", c=colors[i])
    # ax.plot(max_df['time'] / logN, mean_theory_long / logN**(ypower), '--', c='k')
    ax.plot(
        max_df["time"] / logN,
        max_df["Mean Max"] / logN ** (ypower),
        label=N,
        c=colors[i],
        alpha=0.5,
    )

# x, y
start_coord = (250, 25)
end_coord = (8, 3 * 10 ** 4)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))
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
    xy=(140, 10),
    c=colors[0],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)
ax.annotate(
    r"$N=10^{300}$",
    xy=(3, 2 * 10 ** 4),
    c=colors[-1],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)

ax.set_ylim([1, 10 ** 5])
ax.set_xlim([10 ** -3, 5 * 10 ** 3])
fig.savefig("MaxMean.pdf", bbox_inches="tight")

"""
Make plot showing env mean
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^{N})$", fontsize=fontsize, labelpad=0)
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
    cdf_df["Mean Quantile"] -= 2
    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)

    mean_theory = theory.quantileMean(Nquad, cdf_df["time"].values)
    mean_theory_long = theory.quantileMeanLongTime(Nquad, cdf_df["time"].values)

    ax.plot(cdf_df["time"] / logN, mean_theory / logN ** (ypower), "--", c=colors[i])
    # ax.plot(cdf_df['time'] / logN, mean_theory_long / logN**(ypower), '--', c='k')
    ax.plot(
        cdf_df["time"] / logN,
        cdf_df["Mean Quantile"] / logN ** (ypower),
        label=N,
        c=colors[i],
        alpha=0.5,
    )

# x, y
start_coord = (250, 25)
end_coord = (8, 3 * 10 ** 4)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))
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
    xy=(140, 10),
    c=colors[0],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)
ax.annotate(
    r"$N=10^{300}$",
    xy=(3, 2 * 10 ** 4),
    c=colors[-1],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)

ax.set_ylim([1, 10 ** 5])
ax.set_xlim([10 ** -3, 5 * 10 ** 3])
fig.savefig("QuantileMean.pdf", bbox_inches="tight")

"""
Make a plot showing different mean values
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^{N})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))]
ypower = 0

N = 85
i = 3

cdf_df, max_df = db1.getMeanVarN(N)
max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

time = np.geomspace(1, 3 * 10 ** 5 * logN, 1000000)
mean_theory = theory.quantileMean(Nquad, time)
mean_theory_long = theory.quantileMeanLongTime(Nquad, time[time > logN])
ax.plot(time / logN, mean_theory / logN ** (ypower), "-.", c="r")
ax.plot(time[time > logN] / logN, mean_theory_long / logN ** (ypower), "--", c="b")
ax.plot(
    cdf_df["time"] / logN,
    (cdf_df["Mean Quantile"] - 2) / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
)

axins = ax.inset_axes([0.53, 0.05, 0.45, 0.45])
axins.plot(
    time / logN, mean_theory / logN ** (ypower), "-.", c="r", label=r"$M_1(N,t)$"
)
axins.plot(
    time[time > logN] / logN,
    mean_theory_long / logN ** (ypower),
    "--",
    c="b",
    label=r"$M_2(N, t)$",
)
axins.plot(
    cdf_df["time"] / logN,
    (cdf_df["Mean Quantile"] - 2) / logN ** (ypower),
    label=None,
    c=colors[i],
    alpha=0.5,
)
mult = 2
axins.set_xlim([0.8, mult * 2])
axins.set_ylim([1.8 * 10 ** 2, mult * 3.5 * 10 ** 2])
axins.set_xscale("log")
axins.set_yscale("log")
axins.xaxis.set_ticklabels([])
axins.yaxis.set_ticklabels([])
leg = axins.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=["r", "b"],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)
ax.indicate_inset_zoom(axins, edgecolor="black")

ax.set_ylim([1, 10 ** 5])
ax.set_xlim([10 ** -3, 5 * 10 ** 3])
fig.savefig("MeanInterpolation.pdf", bbox_inches="tight")

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

N = 85
i = 3
cdf_df, _ = db1.getMeanVarN(N)

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
var = theory.quantileVar(
    Nquad, cdf_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
)

center = logN ** (1 / 2)
width = 1.3 * logN ** (4 / 3) / logN


ax2 = fig.add_axes([0.53, 0.13, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim([center - width, center + width])
ax2.set_ylim([3 * 10 ** 1, 9 * 10 ** 1])

ax.plot(cdf_df["time"] / logN, var_theory / logN ** (ypower), "--", c="r")
ax.plot(cdf_df["time"] / logN, var_long / logN ** ypower, "--", c="b")
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"] / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
    zorder=0,
)
ax.plot(cdf_df["time"] / logN, var / logN ** (ypower), "--", c="k", alpha=0.5, zorder=1)
xs = np.array([3 * 10 ** 2, 7 * 10 ** 3])
ys = 13 * xs ** (1 / 3)
ys2 = 19 * xs ** (1 / 2)
ax.plot(xs, ys, c="r")
ax.plot(xs, ys2, c="b")
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


ax2.plot(cdf_df["time"] / logN, var_theory / logN ** (ypower), "--", c="r")
ax2.plot(cdf_df["time"] / logN, var_long / logN ** ypower, "--", c="b")
ax2.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"] / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
    zorder=0,
)
ax2.plot(
    cdf_df["time"] / logN, var / logN ** (ypower), "--", c="k", alpha=0.7, zorder=1
)

ax.set_xlim([0.3, 10 ** 4])
ax.set_ylim([10 ** -1, 10 ** 4])
ax.indicate_inset_zoom(ax2, edgecolor="black")

fig.savefig("Interpolation.pdf", bbox_inches="tight")

"""
Make figure plot showing environmental recovery
"""
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
    env_time, env_recovered = theory.log_moving_average(max_df['time'].values, env_subtracted, N, window_size=10**(1/decade_scaling))

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

ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("QuantileVar.pdf", bbox_inches="tight")
