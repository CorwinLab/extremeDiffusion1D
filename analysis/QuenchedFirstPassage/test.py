import numpy as np
import npquad 
from pyDiffusion import DiffusionPDF
from matplotlib import pyplot as plt

maxTime = int(10000 / 2)
beta = 1
dif = DiffusionPDF(1e24, beta, maxTime, True, True)

dif.evolveToTime(maxTime)
occ = dif.occupancy
xvals = (np.arange(0, maxTime+1) - maxTime / 2) * 2

fig, ax = plt.subplots()
ax.scatter(xvals, occ)
ax.set_yscale("log")
ax.set_ylim([1, 1e24])
fig.savefig("Occ.png")

'''
fig, ax = plt.subplots()
ax.scatter(xvals, dif.getTransitionProbabilities()[1:-1])
fig.savefig("Probs.png")
'''

maxTime = 500
beta = 1
dif = DiffusionPDF(1e24, beta, maxTime, True, True)

for i in range(maxTime):
    dif.iterateTimestep()

occ = dif.occupancy
xvals = (np.arange(0, maxTime+1) - maxTime / 2) * 2
print(min(xvals), max(xvals))

fig, ax = plt.subplots()
ax.scatter(xvals, occ)
ax.set_yscale("log")
ax.set_ylim([1, 1e24])
fig.savefig("Occ2.png")

'''
fig, ax = plt.subplots()
ax.hist(np.array(dif.getTransitionProbabilities()[1:-1]).astype(float), bins=100)
ax.set_xlim([0, 1])
ax.set_ylim([0, 25])
fig.savefig("TransProb.png")
'''