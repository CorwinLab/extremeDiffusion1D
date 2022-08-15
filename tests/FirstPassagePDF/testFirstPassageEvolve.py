from pyDiffusion import FirstPassageEvolve, FirstPassageDriver
import numpy as np
import npquad 

beta = 1
maxPositions = [100, 150, 200, 500, 1000]
nParticles = np.quad("1e24")
pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
pdf.setBetaSeed(0)
pdf.evolveToCutoff('TestEvolve.txt')

pdf = FirstPassageDriver(beta, maxPositions)
pdf.setBetaSeed(0)
pdf.evolveToCutoff(nParticles, 'TestDriver.txt')