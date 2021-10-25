import numpy as np
import npquad
from matplotlib import pyplot as plt

Einstein100 = np.loadtxt("/home/jacob/Desktop/corwinLabMount/CleanData/Einstein100/Quartiles0.txt", delimiter=',', skiprows=1)
time = Einstein100[:, 0]
quantile = Einstein100[:, 1]
variance = Einstein100[:, 2]
nParticles = 1e100

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Time / log2(N)")
ax2.set_ylabel("Variance")
ax2.set_ylim([10**-4, 10**5])

fig, ax = plt.subplots()
ax.plot(time / np.log2(nParticles).astype(float), variance)
ax2.plot(time / np.log2(nParticles).astype(float), variance, label='1e100')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-4, max(variance)])
fig.savefig("Variance100.png")


Einstein300 = np.loadtxt("/home/jacob/Desktop/corwinLabMount/CleanData/Einstein300/Quartiles0.txt", delimiter=',', skiprows=1)
time = Einstein300[:, 0]
quantile = Einstein300[:, 1]
variance = Einstein300[:, 2]
nParticles = 1e300

fig, ax = plt.subplots()
ax.plot(time / np.log2(nParticles).astype(float), variance)
ax2.plot(time / np.log2(nParticles).astype(float), variance, label='1e300')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-4, max(variance)])
fig.savefig("Variance300.png")

ax2.legend()
fig2.savefig("VarianceBoth.png")
