import sys
sys.path.append("../src")
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad
from matplotlib import pyplot as plt

beta = np.inf
time = 1000
nParticles = 1e200
d = DiffusionTimeCDF(beta, time)

gumbel = []
for _ in range(time):
    d.iterateTimeStep()
    gumbel.append(d.getGumbelVariance(nParticles))

fig, ax = plt.subplots()
ax.plot(range(time) / np.log2(nParticles), gumbel)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True)
ax.set_ylim([10**-4, max(gumbel)])
ax.set_xlim([0.5, max(range(time) / np.log2(nParticles))])
fig.savefig("Var.png")
