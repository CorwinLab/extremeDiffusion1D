import numpy as np
from pyDiffusion import pyfirstPassageNumba
from matplotlib import pyplot as plt
import time
import copy
from matplotlib import colors
import matplotlib

''' Get spatial prob. distribution over time
x = 500
xvals = np.arange(-x, x+1)
xvals = xvals[1:-1:2]
xvals = np.insert(xvals, 0, xvals[0])
xvals = np.append(xvals, xvals[-1])
pdf = pyfirstPassageNumba.initializePDF(x)

firstPassageCDF = [0]
t=0
while firstPassageCDF[-1] < 0.1:
    pdf = pyfirstPassageNumba.iteratePDF(pdf)
    firstPassageCDF.append(pdf[0] + pdf[-1])

    if t % 1000 == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        bulk_pdf = pdf[1:-1:2]
        bulk_pdf = np.insert(bulk_pdf, 0, pdf[0])
        bulk_pdf = np.append(bulk_pdf, pdf[-1])
        ax.plot(xvals, bulk_pdf)
        ax.set_yscale("log")
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r"$\mathrm{Probability Density}$")
        ax.set_ylim([10**-9, 10**-1])
        fig.savefig(f"./PDFs/RWRE/PDF{t}.png")
        plt.close(fig)

    t+=1
'''

''' Make heatmap
x = 50
tMax = 500
pdf = pyfirstPassageNumba.initializePDF(x)
pdfs = []
for t in range(tMax):
    pdf = pyfirstPassageNumba.iteratePDF(pdf)
    bulk_pdf = pdf[1:-1:2]
    bulk_pdf = np.insert(bulk_pdf, 0, pdf[0])
    bulk_pdf = np.append(bulk_pdf, pdf[-1])
    if ~t%2:
        pdfs.append(bulk_pdf)

cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = 0.2
vmin = 0.00000001

fig, ax =plt.subplots()
ax.imshow(pdfs, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, interpolation="none")
ax.set_xlabel("Distance")
ax.set_ylabel("Time")
ax.set_xlim([0, 50])
ticks = ax.get_xticks()
ax.set_xticklabels([-50, 50])
fig.savefig("PDF.png", bbox_inches='tight')
'''