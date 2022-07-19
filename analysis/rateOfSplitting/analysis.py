import sys
sys.path.append("../src")
from pydiffusionPDF import DiffusionPDF

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
beta = 0.1
d = DiffusionPDF(N, beta, numSteps, ProbDistFlag=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ

allOcc = allOcc[:-2, :]

# Plot the raw Occupancy
color = 'tab:red'
cmap = copy.copy(matplotlib.cm.get_cmap('rainbow'))
cmap.set_under(color='white')
cmap.set_bad(color='white')
vmax = N
vmin = 0.00001

fontsize=12
alpha = 0.3
alpha_line=0.8
#fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
fig, (ax, ax1, ax2) = plt.subplots(figsize=(8,8), nrows=3, constrained_layout=True)
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=1, vmax=vmax), cmap=cmap, interpolation='none')
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
fig.colorbar(cax, ax=ax)

number_of_particles = np.count_nonzero(allOcc, axis=1)
ax1.plot(range(allOcc.shape[0]), number_of_particles)
ax1.set_xlim([1, allOcc.shape[0]])
ax1.set_ylim([1, max(number_of_particles)])
ax1.set_xlabel("t")
ax1.set_ylabel("Number of Particles")

rate_of_particles = np.diff(number_of_particles)
time = range(allOcc.shape[0])[:-1]

def weighted_avg(nPoints, vals):
    return np.convolve(vals, np.ones(nPoints)/nPoints, mode='valid')

nPoints = 1
avg_time = weighted_avg(nPoints, time)
avg_rate = weighted_avg(nPoints, rate_of_particles)
print(avg_time)
ax2.plot(avg_time, avg_rate)

fig.savefig("BetaOccupation.pdf", bbox_inches='tight')
print(np.mean(avg_rate))
''' Now want to check the number of particles at each timestep
allocc = (time, distance)
'''
