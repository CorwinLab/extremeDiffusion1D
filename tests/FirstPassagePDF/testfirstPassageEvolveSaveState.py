import numpy as np
import npquad
from pyDiffusion import FirstPassageEvolve
import os

nParticles = np.quad("1e24")
maxPositions = [500, 1000]
beta = 1
pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
pdf.evolveToCutoff('test.csv')
os.remove("test.csv")
pdf.saveState()

pdf2 = FirstPassageEvolve.fromFile("Scalars.json")
print(pdf2 == pdf)