import numpy as np
from pyDiffusion import pydiffusion2D
from matplotlib import pyplot as plt

nParticles=10000
maxTime = 10000
D = 1
rc = 1
sigma=1
positions = np.zeros(shape=(nParticles))

for t in range(maxTime):
    positions = pydiffusion2D.iterateTimeStep1D(positions, rc, D, sigma)
    print(t)

def gaussuianPDF(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x - mean)**2 / var)

x = np.linspace(min(positions), max(positions), 1000)
pdf2 = gaussuianPDF(x, 0, var=2*D*t)
Dr = D + sigma / rc**2 / np.sqrt(2*np.pi)
pdf = gaussuianPDF(x, 0, var = 2 * Dr * t)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel("Probability Density")
ax.hist(positions, bins=100, density=True)
ax.plot(x, pdf, ls='--', c='k')
ax.plot(x, pdf2, ls='--', c='r')
print("True Variance: ", np.var(positions))
print("Unrenormalized Variance: ", 2 * D * t)
print("Renormalized Variance: ", 2 * Dr * t)
fig.savefig("ParticlePosition.pdf", bbox_inches='tight')
