import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff
import time

N = 1e20
logN = np.log(N)
num_of_steps = logN ** (5/2)
num_of_steps = round(num_of_steps)
print(num_of_steps)
occ = np.zeros(num_of_steps)
occ[0] = int(N)

steps = np.arange(1, num_of_steps+1) * 0.5
start = time.time()

d = cdiff.Diffusion(int(N), 1)
d.setOccupancy(occ)
d.evolveTimesteps(num_of_steps)

minEdges, maxEdges = d.getEdges()
minDistance = abs(minEdges[1:] - steps)
maxDistance = abs(maxEdges[1:] - steps)
meanDistance = (minDistance + maxDistance) / 2

fig, ax = plt.subplots()
ax.plot(steps, meanDistance)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('test.png')
print('Total Time:', time.time() - start)
