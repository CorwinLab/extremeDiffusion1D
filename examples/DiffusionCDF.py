import sys
sys.path.append("../src/")
sys.path.append("../tests/DiffusionCDF")
from nativePyDiffusionCDF import makeRec, findQuintile
from pydiffusionCDF import DiffusionPositionCDF, DiffusionTimeCDF
from matplotlib import pyplot as plt
from theory import theoreticalNthQuart
import numpy as np

tMax = 5000
quantile = 100000
d = DiffusionPositionCDF(1, tMax, [quantile])
d.evolveToPosition(tMax)

timeQuantile = []
dTime = DiffusionTimeCDF(1, tMax)
while dTime.time < tMax:
    timeQuantile.append(dTime.findQuantile(quantile))
    dTime.iterateTimeStep()

CDF = makeRec(tMax)
qs = findQuintile(CDF, quantile).astype(int)

fig, ax = plt.subplots()
ax.plot(np.arange(1, d.tMax+2) / np.log(quantile), d.quantilePositions[0], label='C++ Position Iteration')
ax.plot(np.arange(1, d.tMax+1) / np.log(quantile), timeQuantile, label='C++ Time Iteration')
ax.plot(np.arange(1, d.tMax+1) / np.log(quantile), qs, label='Eric Code')
ax.plot(np.arange(0, d.tMax) / np.log(quantile), theoreticalNthQuart(quantile, np.arange(0, d.tMax)), label='Theory')
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Quantile")
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Mean.png")
