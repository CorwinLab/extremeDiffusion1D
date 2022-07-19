from matplotlib import pyplot as plt
import numpy as np 
import npquad
import sys
sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
import copy
import matplotlib 
from matplotlib import colors

maxTime = 400
maxPosition = 100
beta = np.inf
pdf = FirstPassagePDF(beta, maxPosition)

passageProbability = np.zeros(shape=(maxTime, maxPosition*2 + 1))
for i in range(maxTime):
    pdf.iterateTimeStep()
    pdf_arr = pdf.getPDF()
    passageProbability[i] = pdf_arr

beta=1
pdf = FirstPassagePDF(beta, maxPosition)
passageProbability1 = np.zeros(shape=(maxTime, maxPosition*2 + 1))
for i in range(maxTime):
    pdf.iterateTimeStep()
    pdf_arr = pdf.getPDF()
    passageProbability1[i] = pdf_arr

cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")

fig, (ax, ax2) = plt.subplots(ncols=2)
cax = ax.imshow(
    np.flipud(passageProbability), norm=colors.LogNorm(vmin=1e-15, vmax=1), cmap=cmap, interpolation="none"
)
cax2 = ax2.imshow(
    np.flipud(passageProbability1), norm=colors.LogNorm(vmin=1e-15, vmax=1), cmap=cmap, interpolation="none"
)
ax.set_xlim([0, 200])
ax2.set_xlim([0, 200])
ax2.set_xlabel("Distance")
ax.set_xlabel("Distance")
ax.set_ylabel("Time")
ax2.set_yticklabels([])
ax.set_yticklabels([0, 400, 350, 300, 250, 200, 150, 100, 50, 0])
ax.set_xticklabels([-100, -50, 0, 50, 100])
ax2.set_xticklabels([-100, -50, 0, 50, 100])

fig.savefig("Occupancy.pdf", bbox_inches='tight')
