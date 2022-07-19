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
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/SmallN/3/"
db.add_directory(directory, dir_type="Max", var_file="variables.json")
# db.calculateMeanVar(directory, verbose=True)

for dir in db.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    print(dir, " Systems:", x["number_of_systems"])

fontsize = 12
alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \logN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)

N = 3
_, max_df = db.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df["Var Max"] = max_df["Var Max"] * 4
max_df["Mean Max"] = max_df["Mean Max"] * 2

w = 1
env_recovered = max_df["Var Max"] - theory.gumbel_var(max_df["time"].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode="valid") / w
env_time = np.convolve(max_df["time"].values, np.ones(w), mode="valid") / w
max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
einsten_color = "tab:purple"

var_theory = theory.quantileVar(
    Nquad, max_df["time"].values, crossover=logN ** (1.5), width=logN ** (4 / 3)
)
max_var_theory = var_theory + theory.gumbel_var(max_df["time"].values, Nquad)
ax.plot(
    max_df["time"] / logN,
    max_df["Var Max"],
    c=max_color,
    alpha=alpha,
    label=r"$Var(Max_t^N)$",
)
ax.plot(max_df["time"] / logN, max_var_theory, c=max_color, ls="--")
ax.plot(
    env_time[env_time > logN] / logN,
    env_recovered[env_time > logN],
    c="tab:orange",
    alpha=alpha,
    label=r"$Var(Max_t^N) - Var(Sam_t^N)$",
)
ax.plot(max_df["time"].values / logN, var_theory, ls="--")

fig.savefig("Var.png")
