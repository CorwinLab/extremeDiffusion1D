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