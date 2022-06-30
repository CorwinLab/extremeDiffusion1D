import numpy as np
import npquad 
from firstPassagePDF import FirstPassagePDF
from matplotlib import pyplot as plt
import copy
import matplotlib
from matplotlib import colors

maxTime = 200
maxPosition = 50
beta = 1
pdf = FirstPassagePDF(beta, maxPosition)

allOcc = np.zeros(shape=(maxTime, len(pdf.getPDF())))
for i in range(maxTime):
    allOcc[i, :] = pdf.getPDF()
    pdf.iterateTimeStep()

cmap = copy.copy(matplotlib.cm.get_cmap('rainbow'))
cmap.set_under(color='white')
cmap.set_bad(color='white')
vmin = 10**-17
vmax = 1

fig, ax = plt.subplots()
cax = ax.imshow(allOcc.T, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, interpolation='none')
ax.set_xlabel("Time")
ax.set_ylabel("Position")
fig.colorbar(cax, ax=ax)
fig.savefig("PDF.pdf", bbox_inches='tight')
