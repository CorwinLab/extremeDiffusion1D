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
d = DiffusionPDF(N, 1, numSteps, ProbDistFlag=False)
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

# Plot the raw Occupancy
color = 'tab:red'
cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
cmap.set_under(color='white')
cmap.set_bad(color='white')
vmax = N
vmin = 0.00001

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, aspect='auto', interpolation='none')
ax.plot(d.time, theory / 2 + max(d.time) / 2, c=color)
ax.plot(d.time, max(d.time)/2 - theory / 2, c=color)
ax.set_ylabel("Distance")
ax.set_yticks(np.linspace(0, allOcc.shape[1], 13))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(new_ticks.astype(int))
ax.set_yticklabels(new_ticks)
#ax.set_ylim([90, 60+150+1])
ax2.plot(d.time, d.maxDistance, label='system', c='k')
ax2.plot(d.time, theory / 2, c=color)
ax2.fill_between(d.time, std_below/2, std_above/2, alpha=0.2, color=color)

n_plots = 5
for _ in range(n_plots):
    d = DiffusionPDF(N, 1, numSteps, ProbDistFlag=False)
    d.evolveToTime(numSteps)
    ax2.plot(d.time, d.maxDistance, label='system', c='k', alpha=0.4)
    assert np.all(np.abs(np.diff(2*d.maxDistance)))
    assert np.all(np.abs(np.diff(2*d.minDistance)))

ax2.set_xlabel(r"$t$")
ax2.set_ylabel("Distance")
ax2.set_ylim([0, 55])
#fig.colorbar(cax, ax=ax, label="Particles")
fig.savefig("Occupation.png", bbox_inches='tight')

'''
Make a figure for the talk
'''
color = 'tab:red'
cmap = copy.copy(matplotlib.cm.get_cmap('rainbow'))
cmap.set_under(color='white')
cmap.set_bad(color='white')
vmax = N
vmin = 0.00001

fig, ax = plt.subplots()
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
ax.set_ylabel("Distance")
ax.set_xlabel("Time")
ax.set_yticks(np.linspace(0, allOcc.shape[1], 13))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(new_ticks.astype(int))
ax.set_yticklabels(new_ticks)
ax.set_ylim([425, 575])
#fig.colorbar(cax, ax=ax, label="Particles")
ax.axis("off")
fig.savefig("OccTalkFigure.png", bbox_inches='tight', dpi=1280)
