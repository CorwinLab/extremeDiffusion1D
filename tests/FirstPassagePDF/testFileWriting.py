import numpy as np
import npquad
from libDiffusion import FirstPassageDriver

N = 1e24
distances = range(50, 100)
beta = 1
pdf = FirstPassageDriver(beta, distances)
quantile, variance, positions = pdf.evolveToCutoff(N, 1, '../test.csv', True)
print(positions, quantile, variance)