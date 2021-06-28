import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

files = glob.glob('/home/jhass2/Data/0.01/1.00e_50/Edges*.txt')
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Max Distance from Origin')
ax.set_title('N=1e50, beta=0.01')
ax.set_xscale('log')
ax.set_yscale('log')

colors = ['r', 'b', 'g']
for i, file in enumerate(files[:3]):
    data = np.loadtxt(file)
    minEdge = data[:, 0]
    maxEdge = data[:, 1]
    center = np.arange(1, len(minEdge) + 1) * 0.5
    minDistance = abs(minEdge - center)
    maxDistance = abs(maxEdge - center)
    ax.plot(center * 2, minDistance, c=colors[i])
    ax.plot(center * 2, maxDistance, c=colors[i])

fig.savefig('./figures/EdgesBeta001.png')
