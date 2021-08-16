import sys

sys.path.append("../src")
sys.path.append("../recurrenceRelation")

from pyrecurrence import Recurrance
from nativePyRecurrenceRelation import makeRec, findQuintile
import numpy as np
import npquad
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

N = 3
numbers_to_test = [10, 100, 1000, 10000]

py_means = []
for n in numbers_to_test:
    times = []
    for _ in range(N):
        start = time.time()
        zB = makeRec(n)
        findQuintile(zB, 100)
        findQuintile(zB, 1000)
        times.append(time.time() - start)
    py_means.append(np.mean(times))

c_means = []
for n in numbers_to_test:
    times = []
    for _ in range(N):
        start = time.time()
        rec = Recurrance(beta=np.inf, tMax=n)
        for _ in range(n):
            rec.iterateTimeStep()
            rec.findQuintiles([100, 1000])
        times.append(time.time() - start)
    c_means.append(np.mean(times))

fig, ax = plt.subplots()
ax.set_xlabel("Maximum Simulation Time")
ax.set_ylabel("Runtime (s)")
ax.scatter(numbers_to_test, py_means, label="Python")
ax.scatter(numbers_to_test, c_means, label="C++")
ax.legend()
ax.grid(True)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("SpeedTest.png")
