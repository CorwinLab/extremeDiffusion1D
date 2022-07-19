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
cdf_path_talapas = "/home/jacob/Desktop/talapasMount/JacobData/SmallCDF"

db.add_directory(cdf_path_talapas, dir_type="Gumbel", read_exp=False)
# db.calculateMeanVar(cdf_path_talapas, verbose=True)
quantiles = db.N(dir_type="Gumbel")[0]

fig, ax = plt.subplots()
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r"$Var(Env_t^N)$")
for q in quantiles[4::2]:
    cdf_df, mean_df = db.getMeanVarN(q, read_exp=False)
    logN = np.log(q)
    ax.plot(cdf_df["time"] / logN, cdf_df["Var Quantile"])
ax.grid(True)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Quantiles.png", bbox_inches="tight")
