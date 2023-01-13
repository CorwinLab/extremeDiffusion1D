import numpy as np
from pyDiffusion import pydiffusion2D
from matplotlib import pyplot as plt

nParticles=100_000
maxTime = 10_000
D = 1
rc = 1
positions = np.zeros(shape=(nParticles))

for t in range(maxTime):
    positions = pydiffusion2D.iterateTimeStep1D(positions, rc, D)
    print(t)

def gaussuianPDF(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x - mean)**2 / var)

x = np.linspace(min(positions), max(positions), 1000)
pdf = gaussuianPDF(x, 0, var = 2 * D * t)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel("Probability Density")
ax.hist(positions, bins=100, density=True)
ax.plot(x, pdf, ls='--', c='k')
fig.savefig("ParticlePosition.pdf", bbox_inches='tight')
