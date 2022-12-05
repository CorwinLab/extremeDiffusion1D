import sys

sys.path.append("../../dataAnalysis")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

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
