import numpy as np 
from matplotlib import pyplot as plt

data = np.loadtxt("FirstPassageCDF1.txt", skiprows=1, delimiter=',')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(data[:, 0] / np.log2(1e24), data[:, 2])
ax.set_xlim([0.5, 50])
ax.grid(True)
xvals = np.arange(1, 50)
yvals = xvals ** 4
ax.plot(xvals, yvals)
fig.savefig("test.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(data[:, 0] / np.log2(1e24), data[:, 1])
xvals = np.arange(1, 50)
yvals = xvals ** 2
ax.plot(xvals, yvals * 100)
fig.savefig("test2.png", bbox_inches='tight')
