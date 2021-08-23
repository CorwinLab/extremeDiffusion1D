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
from quadMath import prettifyQuad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/1.0/1Large/Q*.txt")
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

print("Maximum time: ", max(db.time))


db.plotMeans(save_dir="./figures/Means/")
db.plotVars(save_dir="./figures/Vars/")
db.plotMeansEvolve(save_dir="./figures/Means/", legend=False)
db.plotVarsEvolve(save_dir="./figures/Vars/", legend=True)


for i, N in enumerate(db.quantiles):
    var = db.var[:, i]
    Nstr = prettifyQuad(N)
    lnN = np.log(N).astype(float)
    thresh = 0.4

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Time - t0 / lnN")
    ax.set_ylabel("Variance")

    t0 = min(np.where(var > thresh)[0])
    t0 = 600
    num_points = -1
    plot_time = (db.time - t0)[:num_points]
    plot_var = var[:num_points]

    ax.plot(plot_time / lnN, plot_var, label=f"t0={t0}, lnN={int(lnN)}")
    ax.plot(plot_time / lnN, plot_time ** (2 / 3), label="t^(2/3)")
    ax.plot(plot_time / lnN, plot_time, label="t")
    ax.set_title(f"N={Nstr}")
    ax.legend()
    ax.grid(True)
    ax.set_ylim([1e-4, 1e5])
    fig.savefig(f"./figures/VarTurnOn/Var{Nstr}.png", bbox_inches="tight")
    plt.close(fig)
