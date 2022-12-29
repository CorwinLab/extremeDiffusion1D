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
    max_df["time"].values, env_subtracted, window_size=10**(1 / decade_scaling)
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
