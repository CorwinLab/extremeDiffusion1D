import sys

sys.path.append("../../src")
import numpy as np
import npquad
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from databases import QuartileDatabase
import glob
import os
from quadMath import prettifyQuad, logarange
import theory as th

files = glob.glob(
    "/home/jacob/Desktop/corwinLabMount/CleanData/QuartilesMillion/Q*.txt"
)
files = files[:300]

"""
for f in files:
    data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=0)
    tmax = data[-1]
    print(f)
    print(tmax)
    if int(tmax) == 1000000:
        clean_files.append(f)
"""

print("Number of files: ", len(files))

db = QuartileDatabase(files)

run_again = False
if not os.path.exists("./Mean.txt") or not os.path.exists("./Var.txt") or run_again:
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Var.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Var.txt")

print("Maximum Time:", max(db.time))
quarts = np.flip(list(logarange(10, 4500, 10, endpoint=True))) * np.quad("1e4500")
db.setNs(quarts)

"""
db.plotMeans(save_dir='./figures/Means/', verbose=True)
db.plotVars(save_dir='./figures/Vars/', verbose=True)
"""

for i, N in enumerate(db.Ns):
    if np.isinf(N):
        continue
    var = db.var[:, i]
    Nstr = prettifyQuad(N)
    print(Nstr)
    lnN = np.log(N).astype(float)
    thresh = 1e-3

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Time - t0 / lnN")
    ax.set_ylabel("Variance")

    t0 = min(np.where(var > thresh)[0])
    t0 = 1598

    num_points = -1
    plot_time = (db.time - lnN)[:num_points]
    plot_var = var[:num_points]
    theory = th.theoreticalNthQuartVar(N, db.time[:num_points])

    ax.plot(plot_time / lnN, plot_var, label=f"t0={t0}, lnN={int(lnN)}")
    ax.plot(plot_time / lnN, plot_time ** (2 / 3), label="t^(2/3)")
    ax.plot(plot_time / lnN, plot_time, label="t")
    ax.plot(plot_time / lnN, theory, label="Theoretical Curve")
    ax.set_title(f"N={Nstr}")
    ax.legend()
    ax.grid(True)
    ax.set_ylim([1e-4, 1e5])
    fig.savefig(f"./figures/VarTurnOn/Var{Nstr}.png", bbox_inches="tight")
    plt.close(fig)
