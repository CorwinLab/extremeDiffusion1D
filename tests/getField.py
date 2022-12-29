import numpy as np
from pyDiffusion import pydiffusion2D
from matplotlib import pyplot as plt

xi = 1
nParticles = int(1e4)
tMax = 250
D = 1/2
positions = np.random.normal(loc=0, scale=2 * D * tMax, size=nParticles)

fig, ax = plt.subplots()
ax.hist(positions, bins=100)
ax.set_yscale("log")
fig.savefig("ProbDensity.png", bbox_inches='tight')

field = pydiffusion2D.generateGCF1D(positions, xi)

fig, ax = plt.subplots()
ax.scatter(positions, field)
ax.set_xlabel("Particle Position")
ax.set_ylabel("Field Magnitude")
ax.set_title(r"$\xi = 5$")
fig.savefig(f"Field{xi}.png", bbox_inches='tight')