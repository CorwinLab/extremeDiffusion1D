import sys
from pyDiffusion import DiffusionPDF
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import copy

N = 100_000
L = 125
numSteps = 1000
numSteps = int(numSteps)
d = DiffusionPDF(N, 'beta', [1, 1], numSteps, ProbDistFlag=False, staticEnvironment=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))

fpt_position = None
fpt_time = None
for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    min_dist, max_dist = 2*np.array(d.edges) - d.currentTime
    if (min_dist <= -L) and fpt_position is None:
        fpt_position = min_dist
        fpt_time = d.currentTime
    if max_dist >= L and fpt_position is None:
        fpt_position = max_dist
        fpt_time = d.currentTime
    allOcc[i, :] = occ

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ

# Plot the raw Occupancy
color = "tab:red"
cmap = copy.copy(matplotlib.cm.get_cmap("Greys_r"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = N
vmin = 0.00001

fontsize = 12
alpha = 0.3
alpha_line = 0.8

fontsize=12
alpha = 0.3
alpha_line=0.8
#fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
fig, ax = plt.subplots(figsize=(8,8))
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
ax.hlines(500 + L/2, 0, 1000, color='g')
ax.hlines(500 - L/2, 0, 1000, color='g')
ax.scatter(fpt_time, fpt_position / 2 + 500, marker='x', c='r', zorder=2, s=100)

pad = 10
ax.annotate(r"$L$", (100, 500 + L/2 + pad/2), fontsize=fontsize, color='g')
ax.annotate(r"$-L$", (100, 500 - L / 2 - pad), fontsize=fontsize, color='g')
ax.annotate(r"$\tau_{\mathrm{Min}}$", (750, 500 + 75), fontsize=fontsize, color='r')


ax.set_ylabel("Distance", fontsize=fontsize)
ax.set_xlabel("Time", fontsize=fontsize)
ax.set_yticks(np.linspace(0, allOcc.shape[1], 41))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(2*new_ticks.astype(int))
ax.set_yticklabels(new_ticks)
dist = 100
ax.set_ylim([(allOcc.shape[1])/2-dist-.1, (allOcc.shape[1])/2 + dist+0.1])
ax.set_xlim([0, 1000])
ratio = .5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
ax.set_yticks([])
ax.set_xticks([])
fig.savefig("Occupation.pdf", bbox_inches='tight')