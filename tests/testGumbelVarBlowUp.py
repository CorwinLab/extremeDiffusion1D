import sys

sys.path.append("../src")
sys.path.append("../DiffusionPDF")
from diffusionPDF import getWholeEinsteinPDF
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad
from matplotlib import pyplot as plt


def samplePDF(pdf, nParticles, xvals):
    cdf = np.cumsum(pdf)
    samplePDF = []
    for i in range(1, len(cdf)):
        cdf_prev = np.exp(-(1 - cdf[i - 1]) * nParticles)
        cdf_curr = np.exp(-(1 - cdf[i]) * nParticles)
        samplePDF.append(cdf_curr - cdf_prev)
    mean = np.sum(xvals[1:] * samplePDF)
    var = np.sum(((xvals[1:] - mean) ** 2) * samplePDF)
    return var


beta = np.inf
time = 1000
nParticles = 100
d = DiffusionTimeCDF(beta, time)

gumbel = []
vars = []
for t in range(time):
    d.iterateTimeStep()
    gumbel.append(d.getGumbelVariance(nParticles))
    einsteinPDF = getWholeEinsteinPDF(d.time)
    xvals = np.arange(-d.time, d.time + 1, 2)
    var = samplePDF(np.array(einsteinPDF).astype(float), nParticles, xvals)
    vars.append(var)

fig, ax = plt.subplots()
ax.scatter(range(time), gumbel, marker="o")
ax.scatter(range(time), vars, marker="^")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True)
fig.savefig("Var.png")

fig, ax = plt.subplots()
ax.scatter(range(time), np.array(gumbel) - np.array(vars))
ax.set_xscale("log")
fig.savefig("residual.png")
