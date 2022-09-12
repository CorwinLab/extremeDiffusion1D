from pyDiffusion import DiffusionTimeCDF, pyfirstPassageNumba
import numpy as np
import npquad 
from matplotlib import pyplot as plt

tMax = 100
d = 25
position_cdf = DiffusionTimeCDF('beta', [np.inf, np.inf], tMax)
position_cdf.evolveToTime(tMax)
xvals = np.arange(-tMax, tMax, 2)
position_pdf = np.diff(1-np.array(position_cdf.CDF).astype(float))
upper_prob = np.sum(position_pdf[xvals >= d])
lower_prob = np.sum(position_pdf[xvals <= -d])
cutoff_pdf = position_pdf[(xvals < d) * (xvals > -d)]
cutoff_xvals = xvals[(xvals < d) * (xvals > -d)]
cutoff_pdf = np.append(cutoff_pdf, upper_prob)
cutoff_pdf = np.insert(cutoff_pdf, 0, lower_prob)
cutoff_xvals = np.append(cutoff_xvals, cutoff_xvals[-1]+1)
cutoff_xvals = np.insert(cutoff_xvals, 0, cutoff_xvals[0]-1)
print(sum(cutoff_pdf))
print(sum(position_pdf))

# Correct FPT 
pdf = pyfirstPassageNumba.initializePDF(d)
t = 0
while t < tMax:
    pdf = pyfirstPassageNumba.iteratePDF(pdf)
    t += 1
print(sum(pdf))
fpt_xvals = np.arange(-d, d+1, 1)

fig, ax = plt.subplots()
ax.plot(xvals, position_pdf)
ax.plot(fpt_xvals[np.nonzero(pdf)], pdf[np.nonzero(pdf)])
ax.plot(cutoff_xvals, cutoff_pdf)
fig.savefig("FPTVisualization.png", bbox_inches='tight')
