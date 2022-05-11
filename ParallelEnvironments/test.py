import numpy as np
from diffusionSystems import AllSystems
import time

tMax = 1000
numSystems = 100
nParticles = 1e10
system = AllSystems(numSystems, 1, tMax, nParticles)

s = time.time()
for _ in range(tMax-1):
    system.iterateTimeStep()
print(time.time() - s)

'''
system = AllSystems(100, float('inf'), 5, 1000000)
system.iterateTimeStep()
print(np.array(system.getOccupancy()))
system.iterateTimeStep()
print(np.array(system.getOccupancy()))
'''