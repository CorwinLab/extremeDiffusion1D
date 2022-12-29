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