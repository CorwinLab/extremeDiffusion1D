import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")

dir = '/home/jacob/Desktop/talapasMount/JacobData/CyclicScattering'
N = 1e5
mean = np.loadtxt(os.path.join(dir, "Mean.txt"))
var = np.loadtxt(os.path.join(dir, "Var.txt"))
time = np.loadtxt(os.path.join(dir, "Time.txt"))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("Mean(Env)")
ax.set_xlim([1, 10000])
ax.plot(time, mean)

fig.savefig("Mean.pdf", bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("t / log(N)")
ax2.set_ylabel("Var(Env)")
ax2.set_xlim([1 / np.log(N), 10000 / np.log(N)])
ax2.plot(time / np.log(N), var)

fig2.savefig("Var.pdf", bbox_inches='tight')
