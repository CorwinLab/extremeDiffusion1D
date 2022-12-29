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