import sys 
import os
import numpy as np
from matplotlib import pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap
sys.path.append("../../dataAnalysis")

from overalldatabase import Database

db = Database()
einstein_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/EinsteinPaper/"
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
cdf_path = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/CDF/"
cdf_path_talapas = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/Paper/"
dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type="Max")

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")

db1 = db.getBetas(1)
for dir in db.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    print(dir, " Systems:", x["number_of_systems"])

quantiles = db1.N(dir_type="Max")

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))]

fig, ax = plt.subplots()
ax.set_xscale("log")
#ax.set_yscale("symlog")
ax.set_xlabel(r"$t/\log(N)$")
ax.set_ylabel(r"$100\cdot\frac{\mathrm{Var}(\mathrm{Max}^N_t) - \mathrm{Var}(\mathrm{Env}^N_t) - \mathrm{Var}(\mathrm{Sam}^N_t)}{\mathrm{Var}(\mathrm{Max}^N_t)}$")
for i, Nexp in enumerate(quantiles):
    cdf_df, max_df = db1.getMeanVarN(Nexp)

    max_df["Var Max"] = max_df["Var Max"] * 4
    max_df["Mean Max"] = max_df["Mean Max"] * 2

    N = float(f"1e{Nexp}")
    logN = np.log(N)
    max_at_cdf = np.interp(cdf_df['time'].values, max_df['time'], max_df['Var Max'])
    ax.scatter(cdf_df['time'] /logN, 100 * (max_at_cdf - (cdf_df['Gumbel Mean Variance'] + cdf_df['Var Quantile'])) / max_at_cdf, color=colors[i], label=f'1e{Nexp}', s=1)
ax.set_xlim([0.5, 5e3])
ax.set_ylim([-50, 50])
ax.legend()
fig.savefig("NumericalFractionalResidual.pdf", bbox_inches='tight')