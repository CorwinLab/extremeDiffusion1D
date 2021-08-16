import sys

sys.path.append("../../recurrenceRelation")
sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from pyrecurrence import Recurrance
from pydiffusion import theoreticalPbMean, theoreticalNthQuart
import numpy as np
import npquad
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

tmax = 10_000
beta = 1
quartiles = [10, 100, 1000, 10_000]
times = range(1, tmax)
rec = Recurrance(beta, tmax)
rec.evolveAndSaveQuartile(times, quartiles, file="Data.txt")
qs = np.loadtxt("Data.txt", skiprows=1, delimiter=",")

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Nth Quartile")

for col, q in enumerate(quartiles):
    data = qs[:, col + 1]
    time = qs[:, 0]
    ax.plot(time, data, label=f"N={q}")

    # Okay what the fukc is going on - how do I calculate the theoretical value?
    theory_data = data[np.where(data <= time)]
    theory_time = time[np.where(data <= time)]

    theoretical_pb = theoreticalPbMean(theory_data / theory_time, theory_time)
    theoretical_N = 1 / np.exp(theoretical_pb)

    ax.plot(theory_time, theoretical_N)

ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("test.png")
