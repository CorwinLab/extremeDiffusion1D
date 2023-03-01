import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion import pyfirstPassageNumba
plt.rcParams.update({'font.size': 15})
np.random.seed(0)
L = 500
tMax = 10000
pdf = pyfirstPassageNumba.initializePDF(L)

for _ in range(tMax):
    pdf = pyfirstPassageNumba.iteratePDF(pdf)

nonzero_vals = pdf != 0
xvals = np.arange(-L, L+1)

pdf2 = pyfirstPassageNumba.initializePDF(L)
for _ in range(tMax):
    pdf2 = pyfirstPassageNumba.iteratePDF(pdf2, model='SSRW')

fontsize=14
fig, ax = plt.subplots()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p_{\bf{B}}(x, t)$")
ax.set_yscale("log")
ax.set_xlim([-515, 515])
ax.set_ylim([10**-9, 4*10**-2])
ax.plot(xvals[nonzero_vals], pdf[nonzero_vals], c='tab:blue')
ax.plot(xvals[nonzero_vals], pdf2[nonzero_vals], c='k')
fig.savefig("ProbabilityDistribution.svg", bbox_inches='tight')