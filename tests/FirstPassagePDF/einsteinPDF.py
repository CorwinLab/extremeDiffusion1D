import numpy as np
import npquad 
from matplotlib import pyplot as plt 
from pyDiffusion import FirstPassageEvolve

maxPosition = 2000
beta = 1
maxTime = 50000
pdf = FirstPassageEvolve(beta, [maxPosition], 1)

for i in range(maxTime):
    pdf.iterateTimeStep()
    print(i)

pdf = pdf.getPDFs()[0].getPDF()
xvals = np.arange(-maxPosition - 2, maxPosition +2, 2)
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(xvals, pdf)
fig.savefig("BetaPDF.png")