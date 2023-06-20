import numpy as np 
import os 
from matplotlib import pyplot as plt
import pandas as pd

file = '/home/jacob/Desktop/talapasMount/JacobData/Binom100/PDF0.txt'
pdf = np.loadtxt(file)
nonzeros = np.nonzero(pdf)[0]
minIdx, maxIdx = nonzeros[0], nonzeros[-1]
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(pdf[minIdx:maxIdx])
fig.savefig("ProbDistribution.pdf", bbox_inches='tight')
