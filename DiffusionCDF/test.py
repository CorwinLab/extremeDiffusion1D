from diffusionCDF import DiffusionTimeCDF
import numpy as np 
import npquad
import matplotlib.pyplot as plt
from matplotlib import cm

tMax = 5000
N = 100
d = DiffusionTimeCDF(0.01, tMax)
quantile = []
time = []
allOcc = []
for _ in range(tMax):
    d.iterateTimeStep()
    quantile.append(d.findQuantile(N))
    time.append(d.getTime())
    allOcc.append(d.getCDF())

allOcc = np.array(allOcc)
fig, ax = plt.subplots()
ax.scatter(time, quantile, s=0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Quantile Position")
fig.savefig("Quantile.png", bbox_inches='tight')

cmap = cm.jet
cmap.set_bad("black", 1.)
fig, ax = plt.subplots()
im = ax.imshow(allOcc.astype(float), cmap=cmap)
fig.colorbar(im)
fig.savefig("Occupancy.png")