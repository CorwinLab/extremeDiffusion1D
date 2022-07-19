import sys

sys.path.append("../src/")
from pydiffusionPDF import DiffusionPDF
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
d = DiffusionPDF(N, 1, numSteps, ProbDistFlag=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ


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
cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = N
vmin = 0.00001

fontsize = 12
alpha = 0.3
alpha_line = 0.8
# fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(
    allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation="none"
)
ax.plot(d.time, theory / 2 + max(d.time) / 2, c=color, alpha=alpha_line, lw=0.75)
ax.plot(d.time, max(d.time) / 2 - theory / 2, c=color, alpha=alpha_line, lw=0.75)
ax.fill_between(
    d.time,
    (max(d.time) + std_above) / 2,
    (max(d.time) + std_below) / 2,
    color=color,
    alpha=alpha,
)
ax.fill_between(
    d.time,
    (max(d.time) - std_above) / 2,
    (max(d.time) - std_below) / 2,
    color=color,
    alpha=alpha,
)
ax.set_ylabel("Distance", fontsize=fontsize)
ax.set_xlabel(r"t", fontsize=fontsize)
ax.set_yticks(np.linspace(0, allOcc.shape[1], 41))
ticks = ax.get_yticks()
new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
new_ticks = list(2 * new_ticks.astype(int))
ax.set_yticklabels(new_ticks)
dist = 100
ax.set_ylim([(allOcc.shape[1]) / 2 - dist - 0.1, (allOcc.shape[1]) / 2 + dist + 0.1])

ax.set_xlim([0, 1000])
ratio = 0.5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
fig.savefig("Occupation.pdf", bbox_inches="tight")

# Just a check to make sure the scaling matches up.
fig, ax = plt.subplots()
ax.plot(d.time, theory)
ax.fill_between(d.time, (theory + np.sqrt(var)), (theory - np.sqrt(var)), alpha=0.5)
fig.savefig("Theory.png")

fontsize=12
alpha = 0.3
alpha_line=0.8
#fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
fig, ax = plt.subplots(figsize=(8,8))
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
ax.plot(d.time, theory / 2 + max(d.time) / 2, c=color, alpha=alpha_line, lw=0.75)
ax.plot(d.time, max(d.time)/2 - theory / 2, c=color, alpha=alpha_line, lw=0.75)
ax.fill_between(d.time, (max(d.time) + std_above)/2, (max(d.time) + std_below)/2, color=color, alpha=alpha)
ax.fill_between(d.time, (max(d.time) - std_above)/2, (max(d.time) - std_below)/2, color=color, alpha=alpha)
ax.set_ylabel("Distance", fontsize=fontsize)
ax.set_xlabel(r"t", fontsize=fontsize)
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
fig.savefig("ROccupation.pdf", bbox_inches='tight')