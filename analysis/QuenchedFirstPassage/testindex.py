import numpy as np
import npquad 
from pyDiffusion import DiffusionPDF
from matplotlib import pyplot as plt

maxTime = 10
beta = 1
dif = DiffusionPDF(1e24, beta, maxTime, True, True)

for _ in range(maxTime):
    dif.iterateTimestep()
    print(dif.getTransitionProbabilities())