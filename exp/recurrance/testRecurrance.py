import sys

sys.path.append("../../recurrenceRelation")
sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from pyrecurrence import Recurrance
from pydiffusion import theoreticalPbMean
import numpy as np
import npquad
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

tmax = 1_000
beta = 1
quartiles = [10, 100, 1000, 10_000]
times = range(1, tmax)
rec = Recurrance(beta, tmax)
rec.evolveAndSaveQuartile(times, quartiles, file="Data.txt")
qs = np.loadtxt("Data.txt", skiprows=1, delimiter=",")

print(qs[:, 1])
theory = theoreticalPbMean(qs[:, 1] / np.array(times), np.array(times))

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Nth Quartile")
ax.plot(times, theory, label="Theory=10")
for col, q in enumerate(quartiles):
    ax.plot(qs[:, 0], qs[:, col + 1], label=str(q))

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("test.png")
