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
beta_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    if beta == 0.01:
        continue
    db.add_directory(path, dir_type='Max')
    #db.calculateMeanVar(path, verbose=True)

betas = db.betas()
print(betas)
N_exp = db.N(dir_type='Max')[0] # Should be all the same beta
N = np.quad(f"1e{N_exp}")
log2N = np.log(N).astype(float) / np.log(2)

fontsize=12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log_2(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}^N_t)$")

cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(betas) / 1) for i in range(len(betas))
]
betas.sort()
for i, b in enumerate(betas):
    dbb = db.getBetas(b)
    _, max_df = dbb.getMeanVarN(N_exp)
    max_df['Mean Max'] *= 2
    max_df['Var Max'] *= 4

    ax.plot(max_df['time'] / log2N, max_df['Var Max'], c=colors[i], alpha=0.9)


start_coord = (40, 1)
end_coord = (1, 10**4)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))+5
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=80, headlength=40, alpha=0.3), zorder=0)
ax.annotate(r"$\beta=10$", xy=(30, .3), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor', fontsize=fontsize)
ax.annotate(r"$\beta=0$", xy=(.5, 10**4), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor', fontsize=fontsize)
ax.set_xlim([0.01, 3*10**3])
ax.set_ylim([10**-1, 10**5])
fig.savefig("MaxSweep.pdf", bbox_inches='tight')
