import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff
sys.path.append('../src')
import diffusion as diff

beta = 1
N = 1000
occ = np.zeros(N)
occ[0] = N
num_of_steps = 1250
print(num_of_steps)
steps = np.arange(1, num_of_steps+1) * 0.5

d = cdiff.Diffusion(N, beta, int(N*10))
d.setOccupancy(occ)
d.evolveTimesteps(num_of_steps)

minEdges, maxEdges = d.getEdges()
minDistance = abs(minEdges[1:] - steps)
maxDistance = abs(maxEdges[1:] - steps)
avgC = (minDistance + maxDistance) / 2

biasFunction = lambda N: diff.betaBias(N, alpha=1, beta=beta)
edges = diff.floatRunFixedTime(num_of_steps, biasFunction, numWalkers=N)
edges = np.abs(edges)
minPyEdges = edges[:, 0]
maxPyEdges = edges[:, 1]
avgPy = (minPyEdges + maxPyEdges) / 2

fig, ax = plt.subplots()
ax.plot(steps * 2, avgPy, label='Python Code')
ax.plot(steps * 2, avgC, label='C++ code')
ax.set_xlabel('Time')
ax.set_ylabel('Average Distance from Origin')
fig.savefig('CvsPy.png')
