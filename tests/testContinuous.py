import numpy as np
from matplotlib import pyplot as plt

nParticles = 10000
maxTime = 10000
D = 5
numSystems = 100
maxPos = np.zeros(shape=(maxTime))
for j in range(numSystems):
    positions = np.zeros(shape=(nParticles))
    for i in range(maxTime):
        positions += np.random.normal(0, np.sqrt(2*D), size=nParticles)
        maxPos[i] += np.max(positions)
maxPos = maxPos / numSystems
time = np.arange(1, maxTime+1)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time, maxPos)
ax.plot(time, np.sqrt(4 * D * np.log(nParticles)*time), ls='--')
ax.set_xlim([1, maxTime-1])
fig.savefig("Max.png", bbox_inches='tight')