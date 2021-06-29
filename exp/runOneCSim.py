import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff
import time

N = 1e25
logN = np.log(N)
num_of_steps = logN ** (5/2)
num_of_steps = round(num_of_steps)
print(num_of_steps)

steps = np.arange(1, num_of_steps) * 0.5
start = time.time()

d = cdiff.Diffusion(int(N), 1)
d.initializeOccupationAndEdges()
d.evolveTimesteps(num_of_steps, inplace=True)

minEdges, maxEdges = d.getEdges()
minDistance = abs(minEdges[1:] - steps)
maxDistance = abs(maxEdges[1:] - steps)
distance = np.max(np.vstack((minDistance, maxDistance)), 0)

np.savetxt('mean.txt', distance)

fig, ax = plt.subplots()
ax.plot(steps, distance)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('var.png')
print('Total Time:', time.time() - start)
