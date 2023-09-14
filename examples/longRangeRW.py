import numpy as np
from pyDiffusion.pymultijumpRW import *

L = 1000
step_size = 11
tMax = L // step_size

# This sets the size to a little bigger than the required
# size to iterate to tMax.
size = tMax * step_size
pdf = np.zeros(size)
pdf[0] = 1
t = 0 

while t < tMax:
	#print(f"t={t}, {np.sum(pdf)}")
	pdf = iterateTimeStep(pdf, t+1, step_size, symmetric=True)
	t += 1

# This gives the correct xvalues for the array
# note that it only works when the step size is 
# odd. If it's even things get messed up.
center = t * (step_size // 2)
xvals = np.arange(0, pdf.size) - center

# Make sure the measurements are correct
xMeasurement = 500
pMeasurement, cdfMeasurement = measurePDFandCDF(pdf, xMeasurement, t, step_size)

N = 1e12
quantile = measureQuantile(pdf, N, t, step_size)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylim([10**-20, 10**-2])
ax.set_xlim([-1500, 1500])
ax.plot(xvals, pdf)
ax.scatter(xMeasurement, pMeasurement, c='k', zorder=3)
ax.vlines(quantile, 10**-20, 1, color='r', ls='--')
ax.set_xlabel("x")
ax.set_ylabel(r"$p_{\bf{B}}(x,t)$")
fig.savefig("DirichletDist.png")

# Test evolve and get quantile and velocity function
tMax = 1000
times = np.unique(np.geomspace(1, tMax, num = 100).astype(int))
step_size = 11
N = 1e12 
v = 1/2 
save_file = 'Quantiles.txt'

evolveAndMeasureQuantileVelocity(times, step_size, N, v, save_file, False)