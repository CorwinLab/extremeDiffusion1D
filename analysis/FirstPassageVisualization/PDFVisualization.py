from matplotlib import pyplot as plt
import numpy as np 
import npquad
import sys
sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
import copy
import matplotlib 
from matplotlib import colors

maxTime = 5
maxPosition = 2
beta = 1
pdf = FirstPassagePDF(beta, maxPosition)

passageProbability = np.zeros(shape=(maxTime, maxPosition*2 + 1))
for i in range(maxTime):
    pdf_arr = pdf.getPDF()
    pdf.iterateTimeStep()
    passageProbability[i] = pdf_arr

beta=1
pdf = FirstPassagePDF(beta, maxPosition)
passageProbability1 = np.zeros(shape=(maxTime, maxPosition*2 + 1))
for i in range(maxTime):
    pdf_arr = pdf.getPDF()
    pdf.iterateTimeStep()
    passageProbability1[i] = pdf_arr

cmap = copy.copy(matplotlib.cm.get_cmap("Reds"))
cmap.set_under(color="white")
cmap.set_bad(color="white")

fig, ax = plt.subplots()
cax = ax.imshow(
    np.flipud(passageProbability), vmin=0.00001, vmax=1, cmap=cmap, interpolation="none"
)

ax.axis("off")

fig.savefig("FirstPassagePDF.png", bbox_inches='tight', dpi=100)