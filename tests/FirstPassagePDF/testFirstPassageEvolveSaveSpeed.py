import numpy as np
import npquad 
from pyDiffusion import FirstPassageEvolve
import time

nParticles = 1e24 
maxPositions = np.unique(np.geomspace(50, 500*np.log(nParticles), 500).astype(int))
beta = 1

pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
start = time.time()
pdf.saveState()
print(time.time() - start)