import sys
from pyDiffusion import DiffusionPDF
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import copy

N = 1_000
numSteps = 500
numSteps = int(numSteps)
distribution = 'beta'
parameters = [1, 1]

d = DiffusionPDF(N, distribution, parameters, numSteps)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ

centering = np.arange(0, numSteps+1)

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((numSteps/2 - centering[i]/2))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ

parameters = [np.inf, np.inf]
d = DiffusionPDF(N, distribution, parameters, numSteps)
allOccE = np.zeros(shape=(numSteps + 1, numSteps + 1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOccE[i, :] = occ

for i in range(allOccE.shape[0]):
    occ = allOccE[i, :]
    idx_shift = int((numSteps/2 - centering[i]/2))
    occ = np.roll(occ, idx_shift)
    allOccE[i, :] = occ

# Plot the raw Occupancy
color = "tab:red"
cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = N
vmin = 0.00001

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
cax = ax.imshow(
    allOcc.T,
    norm=colors.LogNorm(vmin=1, vmax=vmax),
    cmap=cmap,
    aspect="auto",
    interpolation="none",
)
ax.set_ylabel("Distance")
ax.set_yticks(np.linspace(0, allOcc.shape[1], 21))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(new_ticks.astype(int))
ax.set_yticklabels(new_ticks)
dist = 50
ax.set_ylim([(allOcc.shape[1]) / 2 - dist - 1, (allOcc.shape[1]) / 2 + dist + 1])
cax = ax2.imshow(
    allOccE.T,
    norm=colors.LogNorm(vmin=1, vmax=vmax),
    cmap=cmap,
    aspect="auto",
    interpolation="none",
)
ax2.set_ylabel("Distance")
ax2.set_yticks(np.linspace(0, allOcc.shape[1], 21))
ticks = ax2.get_yticks()
new_ticks = np.linspace(0, allOccE.shape[1], len(ticks)) - (allOccE.shape[1]) / 2
new_ticks = list(new_ticks.astype(int))
ax2.set_yticklabels(new_ticks)
ax2.set_ylim([(allOcc.shape[1]) / 2 - dist - 1, (allOcc.shape[1]) / 2 + dist + 1])
ax2.set_xlabel("Time")

fig.savefig("Comparison0.5.pdf", bbox_inches="tight")
