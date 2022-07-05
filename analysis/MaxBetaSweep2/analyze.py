import glob
from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys
import os

sys.path.append("../../src")
from databases import QuartileDatabase, CDFVarianceDatabase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

data_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/*"
einstein_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmall/Quartiles0.txt"
data_dirs = glob.glob(data_dir)
number_of_plots = 5
fontsize = 12
save_dir = "./Data"

fig, ax = plt.subplots()
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / number_of_plots / 1) for i in range(number_of_plots)]

count = 0
for i, dir in enumerate(data_dirs):
    files = glob.glob(os.path.join(dir, "Q*.txt"))
    beta = dir.split("/")[-1]
    if float(beta) not in [0, 0.05, 0.5, 5]:
        continue
    db = QuartileDatabase(files)
    if os.path.exists(f"Time{beta}.txt"):
        db.time = np.loadtxt(f"Time{beta}.txt")
        db.maxVar = np.loadtxt(f"Var{beta}.txt")
    else:
        db.calculateMeanVar(verbose=True, maxTime=300_000, doubleMax=True)
        np.savetxt(f"Time{beta}.txt", db.time)
        np.savetxt(f"Var{beta}.txt", db.maxVar)
    N = np.quad("1e20")
    logN = np.log2(1e20).astype(float)
    ax.plot(db.time / logN, db.maxVar, label=beta, c=colors[count + 1])
    count += 1

db = CDFVarianceDatabase([einstein_dir])
db.calculateMeanVar()
for i, var in db.quantiles:
    if np.quad(var) == np.quad("1e20"):
        einstein_var = db.gumbelMean[:, i - 1]
        logN = np.log2(1e20).astype(float)
        np.savetxt("Timeinf.txt", db.time)
        np.savetxt("Varinf.txt", einstein_var)
        ax.plot(db.time / logN, einstein_var, c=colors[-1], label=r"$\infty$")

start_coord = (20, 1)
end_coord = (2.5, 2 * 10 ** 3)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx)) + 10
ax.annotate(
    "",
    xy=start_coord,
    xytext=end_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=50,
        headwidth=80,
        headlength=40,
        alpha=0.3,
    ),
    zorder=0,
)
ax.annotate(
    r"$\beta=\infty$",
    xy=(12.5, 0.4),
    c=colors[-1],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)
ax.annotate(
    r"$\beta=0$",
    xy=(1.5, 2 * 10 ** 3),
    c=colors[0],
    rotation=90 - abs(theta),
    rotation_mode="anchor",
    fontsize=fontsize,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log_2(N)$", fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}_t^{(N)})$", fontsize=fontsize)
ax.set_ylim([10 ** -2, 10 ** 5])
ax.set_xlim([0.05, 4 * 10 ** 3])
fig.savefig("Var.png")
