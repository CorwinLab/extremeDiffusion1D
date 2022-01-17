import sys

sys.path.append("../../src")
from databases import VelocityDatabase
import glob
import numpy as np
import npquad
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import theory

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/VelocitySweep/*.txt")
files = files[:5000]  # I'm not sure if all the datasets go out to the final time
db = VelocityDatabase(files)

run_again = False

if not os.path.exists("./Mean.txt") or not os.path.exists("./Vars.txt") or run_again:
    print("Calculating Mean and Variance")
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Vars.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Vars.txt")
print("Number of files:", len(files))


fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")

vs = db.getVelocities()
mean = db.mean
time = db.time
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(vs) / 1) for i in range(len(vs))
]

downsample = 50
for i, v in enumerate(vs):
    log_P = mean[:, i]
    I = theory.I(v)
    sigma = theory.sigma(v)
    val = (log_P + I*time) / time**(1/3) / sigma
    ax.scatter(time[::downsample], abs((val[::downsample]/theory.TW_mean)), label=f'{v}', color=colors[i])
fig.savefig("Collapse.png")
