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

data_dir = '/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/*'
einstein_dir = '/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmall/Quartiles0.txt'
data_dirs = glob.glob(data_dir)
number_of_plots = len(data_dirs) + 1

save_dir = './Data'

fig, ax = plt.subplots()
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / number_of_plots / 1) for i in range(number_of_plots)
]

for i, dir in enumerate(data_dirs):
    files = glob.glob(os.path.join(dir, "Q*.txt"))
    beta = dir.split("/")[-1]
    db = QuartileDatabase(files)
    db.calculateMeanVar(verbose=True, maxTime=300_000, doubleMax=True)
    N = np.quad("1e20")
    logN = np.log(N).astype(float)
    ax.plot(db.time, db.maxVar, label=beta, c=colors[i+1])

db = CDFVarianceDatabase([einstein_dir])
db.calculateMeanVar()
for i, var in db.quantiles:
    if np.quad(var) == np.quad("1e20"):
        einstein_var = db.gumbelMean[:, i-1]
        ax.plot(db.time, einstein_var, c=colors[-1], label=r'$\infty$')

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("Max Particle Variance")
ax.set_ylim([10**-4, 10**4])
ax.set_xlim([1, 300_000])
fig.savefig("Var.png")
