import sys

sys.path.append("../../dataAnalysis/")
from pyDiffusion import DiffusionPDF
from theory import quantileMean, quantileVar, gumbel_var
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import copy

N = 100_000
numSteps = 1000
numSteps = int(numSteps)
d = DiffusionPDF(N, 'beta', [1,1], numSteps, ProbDistFlag=False, staticEnvironment=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))
maxLoc = np.zeros(shape=numSteps + 1)
minLoc = np.zeros(shape=numSteps + 1)

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ
    maxLoc[i] = d.getMaxIdx() - d.currentTime / 2
    minLoc[i] = d.getMinIdx() - d.currentTime / 2

theory = quantileMean(N, d.time)
var = quantileVar(N, d.time) + gumbel_var(d.time, N)
num_of_std = 1
std_below = theory - num_of_std * np.sqrt(var)
std_above = theory + num_of_std * np.sqrt(var)

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

fontsize=12
alpha = 0.3
alpha_line=0.8
fig, (ax, ax2) = plt.subplots(figsize=(8,8), nrows=2, ncols=1, sharex=True)

cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
ax.plot(d.time[:-1], maxLoc[:-1] + max(d.time[:-1]) / 2, c='g')
ax.plot(d.time, theory / 2 + max(d.time) / 2, c=color, alpha=alpha_line, lw=0.75)
ax.plot(d.time, max(d.time)/2 - theory / 2, c=color, alpha=alpha_line, lw=0.75)
ax.fill_between(d.time, (max(d.time) + std_above)/2, (max(d.time) + std_below)/2, color=color, alpha=alpha)
ax.fill_between(d.time, (max(d.time) - std_above)/2, (max(d.time) - std_below)/2, color=color, alpha=alpha)

ax.annotate(r"$\mathrm{Max}_{t}^{N}$", xy=(250, 500+60), c='g', fontsize=fontsize)

ax.set_ylabel("Distance", fontsize=fontsize)

'''
ax.set_yticks(np.linspace(0, allOcc.shape[1], 41))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(2*new_ticks.astype(int))
'''
ax.set_yticklabels([])
ax.set_xticklabels([])

dist = 100
ax.set_ylim([(allOcc.shape[1])/2-dist-.1, (allOcc.shape[1])/2 + dist+0.1])
ax.set_xlim([0, 1000])

ratio = .5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

cax = ax2.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
L = 125
ax2.hlines(y=numSteps/2 + L / 2, xmin=0, xmax=1000, ls='--', color='b')
ax2.hlines(y=numSteps/2 - L / 2, xmin=0, xmax=1000, ls='--', color='b')

maxIdx, minIdx = np.argmax(2*maxLoc == L), np.argmax(2 * minLoc == -L)
if maxIdx < minIdx:
    ax2.scatter(d.time[maxIdx], maxLoc[maxIdx] + max(d.time[:-1]) / 2, c='r', marker='*', zorder=np.inf)
else:
    ax2.scatter(d.time[minIdx], minLoc[minIdx] + max(d.time[:-1]) / 2, c='r', marker='*', zorder=np.inf)

ax2.set_ylabel("Distance", fontsize=fontsize)
ax2.set_xlabel("Time", fontsize=fontsize)
ax2.annotate("L", xy=(750, 500 + L /2 + 10), fontsize=fontsize, c='blue')
ax2.annotate("-L", xy=(750, 500 - L /2 - 15), fontsize=fontsize, c='blue')
ax2.annotate(r"$\tau_{\mathrm{Min}}$", xy=(250, 500 - L /2 - 15), c='red', fontsize=fontsize)

'''
ax2.set_yticks(np.linspace(0, allOcc.shape[1], 41))
ticks = ax2.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(2*new_ticks.astype(int))
'''

ax2.set_yticklabels([])
ax2.set_xticklabels([])

dist = 100
ax2.set_ylim([(allOcc.shape[1])/2-dist-.1, (allOcc.shape[1])/2 + dist+0.1])
ax2.set_xlim([0, 1000])

ratio = .5
x_left, x_right = ax2.get_xlim()
y_low, y_high = ax2.get_ylim()
ax2.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

fig.savefig("Occupation.pdf", bbox_inches='tight')