import sys

sys.path.append("../cDiffusion")
from cDiffusion import Diffusion
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

N = 1e25
Nquart = 1 / N
num_of_steps = 3 * np.log(N) ** (5 / 2)
record_times = np.geomspace(1, num_of_steps, 10000, dtype=np.int64)
record_times = np.unique(record_times)
dts = np.diff(record_times)
quartiles = []

d = Diffusion(1.0, 1.0, smallCutoff=0, largeCutoff=0)
d.initializeOccupationAndEdges(record_times[-1])
for t in dts:
    d.evolveTimesteps(t, inplace=True)
    quartiles.append(d.getNthquartile(Nquart))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Nth Quartile")
ax.set_title("alpha=beta=1.0")
ax.plot(record_times[1:], quartiles, c="k", label="N=1e25")
ax.legend()
fig.savefig("NthQuart.png")
