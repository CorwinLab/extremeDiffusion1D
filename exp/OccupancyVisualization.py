import sys
sys.path.append('../src/')
from pydiffusion import Diffusion
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

N = 10000
numSteps = np.log(N) ** (5/2)
numSteps = int(numSteps)
d = Diffusion(N, 1, numSteps)
allOcc = np.zeros(shape=(numSteps+1, numSteps+1))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ

# Plot the Occupancy with a threshold to highlight outliers
threshold = 2
greater = (allOcc * (allOcc > threshold))
less = (allOcc * (allOcc <= threshold))

alpha = (less > 0).astype(float)
vmax = N / 10

fig, ax = plt.subplots()
cax = ax.imshow(greater, cmap='gist_heat_r', vmin=0, vmax=vmax)
ax.imshow(less>0, cmap='Blues', alpha=alpha, label='Positions w/ <2 Particles')
ax.plot(d.center, range(allOcc.shape[0]), 'g--', label='Center')
ax.set_ylabel('Time')
ax.set_xlabel('Distance')
ax.legend()
fig.colorbar(cax, ax=ax, label='Number of Particles')
fig.savefig('Occupation.png')

NthQuart = N
times = d.center * 2
theory = np.piecewise(times, [times < np.log(NthQuart), times >= np.log(NthQuart)], [lambda x: x, lambda x: x*np.sqrt(1-(1-np.log(NthQuart)/x)**2)])
theory = theory / 2

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ

# Plot the raw Occupancy
fig, ax = plt.subplots()
cax = ax.imshow(allOcc, cmap='gist_heat_r', vmin=0, vmax=vmax)
ax.plot([max(d.center), max(d.center)], [0, allOcc.shape[0]], c='g', ls='--', label='Center')
ax.plot(theory+max(d.center), times, c='b', ls='--', label='Theoretical\nMaximum Particle')
ax.plot(max(d.center) - theory, times, c='b', ls='--')
ax.set_ylabel('Time')
ax.set_xlabel('Distance')
ax.legend(fontsize=8, loc='lower left')
ax.set_ylim([0, allOcc.shape[0]])
fig.colorbar(cax, ax=ax, label='Number of Particles')
fig.savefig('Occupation_No_Threshold.png')
