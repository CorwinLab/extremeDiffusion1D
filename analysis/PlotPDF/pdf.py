import sys
sys.path.append("../../src")

from pydiffusionPDF import DiffusionPDF
import numpy as np
import npquad
from matplotlib import pyplot as plt

einstein = DiffusionPDF(1, float("inf"), 100, True)
einstein.evolveToTime(100)
pdf = einstein.occupancy
xvals = np.arange(-len(pdf) / 2, len(pdf) / 2, 1)

fig, ax = plt.subplots()
ax.bar(xvals, pdf)
ax.set_xlim([-50, 50])
#ax.set_ylim([0.001, 1.05*max(pdf)])
ax.set_xlabel("Distance")
ax.set_ylabel("Probability Density")
ax.set_yscale("log")
fig.savefig("Einstein.png", bbox_inches='tight')

beta = DiffusionPDF(1, 1, 100, True)
beta.evolveToTime(100)
pdf = beta.occupancy
xvals = np.arange(-len(pdf) / 2, len(pdf) / 2, 1)

fig, ax = plt.subplots()
ax.bar(xvals, pdf)
ax.set_xlim([-50, 50])
#ax.set_ylim([0, 1.05*max(pdf)])
ax.set_xlabel("Distance")
ax.set_ylabel("Probability Density")
ax.set_yscale("log")
fig.savefig("Beta.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(xvals, np.cumsum(pdf))
ax.set_xlim([-50, 50])
ax.set_xlabel("Distance")
ax.set_ylabel("Probability")
fig.savefig("BetaCDF.png", bbox_inches='tight')
