import numpy as np
import os 
from matplotlib import pyplot as plt

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweepLonger'
betas = os.listdir(dir)
betas.sort()
N = 1e5

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$Var(Env^N_{\infty})$")

for b in betas: 
	if float(b) not in [0.1, 0.01, 1]:
		continue
	mean = np.loadtxt(os.path.join(dir, b, "Mean.txt"))
	var = np.loadtxt(os.path.join(dir, b, "Var.txt"))
	time = np.loadtxt(os.path.join(dir, b, "Time.txt"))
	
	b = float(b)
	beta_var = b ** 2 / (2 * b)**2 / (2 * b + 1)

	ax.scatter(b, var[-1], c='b')

xvals = np.array([10**-2, 1])
yvals = xvals ** -(1)

ax.plot(xvals, yvals*1.2, ls='--', c='k')

fig.savefig("Beta.png", bbox_inches='tight')