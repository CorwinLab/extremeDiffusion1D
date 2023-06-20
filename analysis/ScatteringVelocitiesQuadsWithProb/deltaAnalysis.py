import numpy as np
from matplotlib import pyplot as plt
import os 
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import binom

def gauss(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(- (x-mean)**2 / 2 / var)

def delta(t, v):
	x = v * t**(3/4)
	return x / t * gauss(x, 0, t)

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocitiesQuads'
vs = os.listdir(dir)
vs = [i for i in vs if float(i) > 0.2]
vs.sort()
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(vs) / 1) for i in range(len(vs))]

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel(r"$\mathbb{E}[\Delta(x=vt^{3/4}, t)]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([10, 10**5])
ax.set_ylim([10**-15, 10])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, mean[:, 2], c=colors[i], label=fr'$v={v}$', alpha=0.5)
	ax.plot(times, 2*delta(times, v), ls='--', c=colors[i])

ax.legend()
fig.savefig("Delta.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel(r"$\frac{P(X>vt^{3/4}, t)}{\Delta(x=vt^{3/4}, t)}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1, 10**5])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, np.exp(mean[:, 1])/mean[:, 2], c=colors[i], label=fr'$v={v}$', alpha=0.5)

	xvals = np.array([10**3, 10**5])
	label = r'$\frac{\sqrt{t}}{2v^2}$'
	if v != 0.9:
		label = None
	ax.plot(xvals, xvals**(1/2) / v**2 / 2, ls='--', c=colors[i], label=label)

ax.legend()
fig.savefig("Ratio.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel(r"$\mathbb{E}[\Delta(x=vt^{3/4}, t)]$")
#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([10, 10**5])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.plot(times, var[:, 2], c=colors[i], label=fr'$v={v}$', alpha=0.5)
	#ax.plot(times, 2*delta(times, v), ls='--', c=colors[i])

ax.legend()
fig.savefig("DeltaVar.pdf", bbox_inches='tight')