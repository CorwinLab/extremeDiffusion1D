import sys

sys.path.append("../src/")
from pydiffusionPDF import DiffusionPDF
from theory import quantileMean, quantileVar
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import copy

N = 100_000000
numSteps = 1000
numSteps = int(numSteps)
d = DiffusionPDF(N, float('inf'), numSteps, ProbDistFlag=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ


theory = quantileMean(N, d.time)
var = quantileVar(N, d.time)
std_below = theory - var
std_above = theory + var

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ

color = "tab:red"
cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = N
vmin = 0.00001

fig, ax = plt.subplots()
cax = ax.imshow(
    allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation="none"
)
ax.set_ylabel("Distance")
ax.set_xlabel("Time")
ax.set_yticks(np.linspace(0, allOcc.shape[1], 13))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(new_ticks.astype(int))
ax.set_yticklabels(new_ticks)

ax.set_ylim([400, 600])
#fig.colorbar(cax, ax=ax, label="Particles")
ax.axis("off")
fig.savefig("OccTalkFigure.png", bbox_inches='tight', dpi=1280)
fig.savefig("OccTalkFigure.pdf", bbox_inches='tight', dpi=1280)