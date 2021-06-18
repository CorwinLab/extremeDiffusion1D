import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff

N = 214748364
logN = np.log(N)
num_of_steps = logN ** (5/2)
num_of_steps = round(num_of_steps)
print(num_of_steps)
occ = np.zeros(num_of_steps)
occ[0] = int(N)
N = int(N)

steps = np.arange(1, num_of_steps+1) * 0.5

print('Running the Algorithm')
d = cdiff.Diffusion(N, 1, int(N*10))
d.setOccupancy(occ)
d.evolveTimesteps(num_of_steps)

print('Got past the C++ code')
minEdges, maxEdges = d.getEdges()
minDistance = abs(minEdges[1:] - steps)
maxDistance = abs(maxEdges[1:] - steps)

fig, ax = plt.subplots()
ax.plot(steps * 2, minDistance, c='b', label='Left Edge')
ax.plot(steps * 2, maxDistance, c='k', label='Right Edge')
ax.set_xlabel('Time')
ax.set_ylabel('Distance from Origin')
ax.set_title(f'Number Particles = {d.getN()}, Beta = {d.getBeta()}')
ax.legend()
fig.savefig('Distances Bigger.png')
