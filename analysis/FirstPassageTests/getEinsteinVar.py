from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys 
sys.path.append("../../src")
from pyfirstPassagePDF import runExperiment

N = 1e24
logN = np.log(N) 
distances = np.unique(np.arange(logN, 350, 2).astype(int))
beta = np.inf
run_again = False
if run_again: 
    mean, var = runExperiment(distances, N, beta, verbose=True) 
    np.savetxt('Data.txt', np.array([distances, mean, var]).T)
else:
    data = np.loadtxt("Data.txt")
    distances = data[:, 0]
    mean = data[:, 1]
    var = data[:, 2]

fig, ax = plt.subplots()
ax.plot(distances, mean)
ax.set_xlim([min(distances), max(distances)])
ax.set_xlabel("Distance")
ax.set_ylabel("Mean(First Passage Time)")
ax.grid(True)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(distances, var)
ax.set_xlim([min(distances), max(distances)])
ax.set_xlabel("Distance")
ax.set_ylabel("Var(First Passage Time)")
ax.grid(True)
ax.set_yscale("log")
ax.set_ylim([10**-4, max(var)])
ax.set_xscale("log")
fig.savefig("Var.png", bbox_inches='tight')