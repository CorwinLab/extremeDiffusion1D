import sys
sys.path.append('../src/')
from cdiffusion import Diffusion
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


N = 1e25
numSteps = np.log(N) ** (5/2)
numSteps = int(numSteps)
d = Diffusion(N, 0.01)
d.initializeOccupation()
allOcc = np.zeros(shape=(numSteps, numSteps))

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    if (np.sum(np.isnan(occ)))> 0:
        print(occ)
        break
    occ = np.array(occ, dtype=np.float64)
    isna = np.sum(np.isnan(occ))
    if isna > 0:
        print(occ)
        print(isna)
        break
    allOcc[i, :] = occ
    if i%100 == 0:
        print(i)

im = allOcc > 0.0

fig, ax = plt.subplots()
ax.imshow(im, cmap='Greys', interpolation='None')
ax.set_ylabel('Time')
ax.set_xlabel('Distance')
fig.savefig('Occupation.png')

'''
minEdge, maxEdge = d.getEdges()
center = np.arange(1, len(minEdge) + 1) * 0.5
minDistance = abs(minEdge - center)
maxDistance = abs(maxEdge - center)

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Distance to Origin')
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(center * 2, minDistance, c='r', label='Minimum')
ax.plot(center * 2, maxDistance, c='k', label='Maximum')
ax.legend()
fig.savefig('test.png')
'''
