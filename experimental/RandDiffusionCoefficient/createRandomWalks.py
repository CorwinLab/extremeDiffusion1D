import numpy as np
from scipy.stats import truncnorm 
from matplotlib import pyplot as plt

D0 = 1
std = 0.1
a = (0 - D0) / std # low bound of truncated Gaussian
tMax = 1000000
dt = 1
nParticles = 50

r = truncnorm.rvs(a, np.inf, loc=D0, scale=std, size=(tMax, nParticles))

rw = np.cumsum(np.sqrt(2* r) * np.random.normal(loc=0, scale=1, size=r.shape), axis=0) * dt

fig, ax = plt.subplots()
for i in range(nParticles):
	ax.plot(np.arange(1, tMax+1), rw[:, i])
ax.set_xlabel("Time")
ax.set_ylabel("Distance")
fig.savefig("RandomWalkers.pdf")