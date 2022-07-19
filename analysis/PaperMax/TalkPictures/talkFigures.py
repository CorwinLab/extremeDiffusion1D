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

"""
Make plot showing quantile variance
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

ax2 = fig.add_axes([0.53, 0.21, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax2.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^{(N)})$", labelpad=0, fontsize=fontsize)
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
    max_df["Var Max"] = max_df["Var Max"] * 4
    max_df["Mean Max"] = max_df["Mean Max"] * 2

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)

    var_theory = theory.quantileVar(
        Nquad, cdf_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
    )
    mean_theory = theory.quantileMean(Nquad, cdf_df["time"].values)

    w = 25
    env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
    env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
    env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w

    ax.plot(cdf_df["time"] / logN, var_theory / logN ** (ypower), "--", c=colors[i])
    ax.plot(
        cdf_df["time"] / logN,
        cdf_df["Var Quantile"] / logN ** (ypower),
        label=N,
        c=colors[i],
        alpha=0.5,
    )
    ax.plot(env_time / logN, env_recovered, c=colors[i])

    ax2.plot(cdf_df["time"] / logN, cdf_df["Mean Quantile"] - 2, c=colors[i], alpha=0.8)
    ax2.plot(cdf_df["time"] / logN, mean_theory, "--", c=colors[i])


# x, y
start_coord = (250, 6)
end_coord = (75, 3 * 10 ** 3)
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
    xy=(155, 2.6),
    c=colors[0],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)
ax.annotate(
    r"$N=10^{300}$",
    xy=(40, 2.5 * 10 ** 3),
    c=colors[-1],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)

ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
ax2.remove()
fig.savefig("QuantileVar.png", bbox_inches="tight")
fig.savefig("./TalkPictures/QuantileVar.png", bbox_inches="tight")


"""
Make a couple of figures for the talk
"""
"""
Short time plot
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

N = 85
i = 3
color = "r"
cdf_df, _ = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df["time"].values
var_theory = np.piecewise(
    time,
    [time < logN, time >= logN],
    [lambda t: 0, lambda t: theory.quantileVarShortTime(Nquad, t)],
)

ax.plot(cdf_df["time"] / logN, var_theory / logN ** (ypower), "--", c=color)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"] / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
)

ax.set_xlim([0.3, 10 ** 4])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/ShortVar.png", bbox_inches="tight")

"""
Data plot
"""
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis="both", labelsize=fontsize)

N = 85
i = 3
cdf_df, _ = db1.getMeanVarN(N)
ypower = 0

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df["time"].values
var_theory = theory.quantileVar(
    Nquad, cdf_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
)

# ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=colors[i])
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"] / logN ** (ypower),
    label=N,
    c=colors[i],
    alpha=0.5,
)

ax.set_xlim([0.3, 10 ** 4])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/ShortLongVar.png", bbox_inches="tight")

"""
SSRW vs. RWRE
"""
fontsize = 12
N = 7
cdf_df, max_df = dbe.getMeanVarN(N)
N = float(f"1e{N}")
logN = np.log(N)

max_df["Var Max"] *= 4
c1 = np.loadtxt("7_c1.dat")
einstein_theory = theory.einstein_var(Nquad, c1)

fig, ax = plt.subplots()

ax.set_xlabel(r"$t/ \log(N)$", fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}_t^N)$", fontsize=fontsize)

ax.plot(max_df["time"] / logN, max_df["Var Max"], label="SSRW")

N = 7
cdf_df, max_df = db1.getMeanVarN(N)
N = float(f"1e{N}")
logN = np.log(N)
max_df["Var Max"] *= 4
ax.plot(max_df["time"] / logN, max_df["Var Max"], label="RWRE", c="r")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
ax.tick_params(axis="both", labelsize=fontsize)
ax.legend(fontsize=fontsize)

fig.savefig("SSRWvsRWRE.png", bbox_inches="tight")

"""
Make a plot of SSRW
"""

N = 7
cdf_df, max_df = dbe.getMeanVarN(N)
N = float(f"1e{N}")
logN = np.log2(N)

max_df["Var Max"] *= 4
c1 = np.loadtxt("7_c1.dat")
einstein_theory = theory.einstein_var(Nquad, c1)

fig, ax = plt.subplots()

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$Var(Max_t^N)$")

ax.scatter(max_df["time"], max_df["Var Max"], label="SSRW")
ax.plot(
    max_df["time"][max_df["time"] > logN.astype(float)],
    einstein_theory,
    c=einsten_color,
    ls="--",
    label="Theory",
)

df_time = max_df[max_df["time"] > logN]
subtracted = df_time["Var Max"] - einstein_theory

w = 25
env_recovered = np.convolve(subtracted, np.ones(w), mode="valid") / w
env_time = np.convolve(df_time["time"].values, np.ones(w), mode="valid") / w

ax.plot(env_time, env_recovered, c="k", label="SSRW - Theory")

ax.legend()
ax.set_xlim([10, 100])
ax.set_ylim([0, 10])
fig.savefig("SSRW.png")

"""
Fig, env plots
"""
fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

w = 50
env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w
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
    label=r"$Var(Max_t^N)$",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"],
    c=quantile_color,
    alpha=alpha,
    label=r"$Var(Env_{t}^N)$",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Gumbel Mean Variance"],
    c=gumbel_color,
    alpha=alpha,
    label=r"$Var(Sam_t^N)$",
)
ax.plot(
    env_time[env_time > logN] / logN,
    env_recovered[env_time > logN],
    c="tab:orange",
    alpha=alpha,
    label=r"$Var(Max_t^N) - Var(Sam_t^N)$",
)

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, quantile_color, gumbel_color, "tab:orange"],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/EnvComp4.png", bbox_inches="tight")

"""
Fig, env plots
"""
fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

w = 50
env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w
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
    label=r"$Var(Max_t^N)$",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"],
    c=quantile_color,
    alpha=alpha,
    label=r"$Var(Env_{t}^N)$",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Gumbel Mean Variance"],
    c=gumbel_color,
    alpha=alpha,
    label=r"$Var(Sam_t^N)$",
)
# ax.plot(env_time[env_time > logN] / logN, env_recovered[env_time > logN], c='tab:orange', alpha=alpha, label=r'$Var(Max_t^N) - Var(Sam_t^N)$')

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, quantile_color, gumbel_color, einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/EnvComp3.png", bbox_inches="tight")

"""
Fig, env plots
"""
fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

w = 50
env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w
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
    label=r"$Var(Max_t^N)$",
)
ax.plot(
    cdf_df["time"] / logN,
    cdf_df["Var Quantile"],
    c=quantile_color,
    alpha=alpha,
    label=r"$Var(Env_{t}^N)$",
)
# ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], c=gumbel_color, alpha=alpha, label=r'$Var(Sam_t^N)$')
# ax.plot(env_time[env_time > logN] / logN, env_recovered[env_time > logN], c='tab:orange', alpha=alpha, label=r'$Var(Max_t^N) - Var(Sam_t^N)$')

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, quantile_color, gumbel_color, einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/EnvComp2.png", bbox_inches="tight")
"""
Fig, env plots
"""
fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

w = 50
env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w
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
    label=r"$Var(Max_t^N)$",
)
# ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], c=quantile_color, alpha=alpha, label=r'$Var(Env_{t}^N)$')
# ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], c=gumbel_color, alpha=alpha, label=r'$Var(Sam_t^N)$')
# ax.plot(env_time[env_time > logN] / logN, env_recovered[env_time > logN], c='tab:orange', alpha=alpha, label=r'$Var(Max_t^N) - Var(Sam_t^N)$')

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, quantile_color, gumbel_color, einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/EnvComp1.png", bbox_inches="tight")

"""
Talk SSRW and Gumbel
"""

fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

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
    cdf_df["time"] / logN,
    cdf_df["Gumbel Mean Variance"],
    c=gumbel_color,
    alpha=alpha,
    label=r"$Var(Sam_t^N)$",
)

cdf_df, max_df = dbe.getMeanVarN(N)
max_df["Var Max"] *= 4
ax.plot(
    max_df["time"] / logN, max_df["Var Max"], alpha=alpha, c=einsten_color, label="SSRW"
)

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[gumbel_color, einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/Gumbel2.png", bbox_inches="tight")

"""
Talk SSRW and Gumbel
"""

fontsize = 12
alpha = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 7
cdf_df, max_df = db1.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

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
    cdf_df["time"] / logN,
    cdf_df["Gumbel Mean Variance"],
    c=gumbel_color,
    alpha=alpha,
    label=r"$Var(Sam_t^N)$",
)

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=[gumbel_color, einsten_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
ax.set_xlim([0.3, 5 * 10 ** 3])
ax.set_ylim([10 ** -1, 10 ** 4])
fig.savefig("./TalkPictures/Gumbel1.png", bbox_inches="tight")
