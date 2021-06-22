import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff
import time

N = 1e300
smallCutoff = int(1e9)
logN = np.log(N)
num_of_steps = logN ** (5/2)
num_of_steps = round(num_of_steps)
print(num_of_steps)
occ = np.zeros(num_of_steps)
occ[0] = int(N)

steps = np.arange(1, num_of_steps+1) * 0.5
start = time.time()

d = cdiff.Diffusion(int(N), 1, smallCutoff)
d.setOccupancy(occ)
d.evolveTimesteps(num_of_steps)

minEdges, maxEdges = d.getEdges()
minDistance = abs(minEdges[1:] - steps)
maxDistance = abs(maxEdges[1:] - steps)
meanDistance = (minDistance + maxDistance) / 2

np.savetxt(f'./meandistance300.txt', meanDistance)
print('Total Time:', time.time() - start)
