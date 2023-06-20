import numpy as np
from matplotlib import pyplot as plt

D = 1
sigma = 0.1
nSystems = 500
v = 1/2
dt = 0.01
nParticles = 100_000
tMax = 1e5
ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))

max_pos = np.zeros(ts.shape)
for i in range(len(ts)):
	num = ts[i] / dt
	var = np.random.normal(loc = num * D, scale = num * sigma, size=nParticles)
	pos = np.sqrt(dt) * np.random.normal(loc=0, scale=np.sqrt(var))
	max_pos[i] = np.max(pos)
	print(i)

fig, ax = plt.subplots()
ax.plot(ts, max_pos)
ax.plot(ts, np.sqrt(4 * D * np.log(nParticles) * ts))
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("MaxPos.pdf", bbox_inches='tight')