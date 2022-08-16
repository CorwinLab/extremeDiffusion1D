import numpy as np 
import npquad 
from pyDiffusion import FirstPassageEvolve

nParticles = np.quad("1e24")
maxPositions = [500, 100]
beta = 1 
pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
pdf.evolveToCutoff("Test.csv")
print(pdf == pdf)