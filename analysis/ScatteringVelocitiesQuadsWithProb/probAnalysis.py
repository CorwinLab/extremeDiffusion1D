import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocitiesQuadsWithProbs'
vs = os.listdir(dir)
vs = [i for i in vs if float(i) > 0.2]
vs.sort()
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(vs) / 1) for i in range(len(vs))]

def gaussian(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x-mean)**2 / var)

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$-\mathbb{E}[\ln(P_{\bf{B}}(X=vt^{3/4}, t)]$")
ax.set_xlim([2, 10**5])
ax.set_ylim([10**-15, 10**1])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, mean[:, 3], c=colors[i], label=fr'$v={v}$', alpha=0.75)
	ax.plot(times, 2*gaussian(v * times**(3/4), 0, times), ls='--', c=colors[i])

ax.legend()
fig.savefig("Mean.pdf")

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\frac{\mathbb{E}[P_{\bf{B}}(X=vt^{3/4}, t)]}{\mathbb{E}[\Delta_{\bf{B}}(X=vt^{3/4}, t)]}$")
ax.set_xlim([2, 10**5])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, mean[:, 3] / mean[:, 2], c=colors[i], label=fr'$v={v}$', alpha=0.5)

	xvals = np.array([10**3, 10**5])
	label = r'$t^{1/4} / v$'
	if v != 0.9:
		label = None 

	ax.plot(xvals, xvals ** (1/4) / v, c=colors[i], ls='--', label=label)

ax.legend()
fig.savefig("PDeltaRatio.pdf")

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$-\frac{\sqrt{t}\log(v)^{1/6}}{v^2}\left(\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t)]\right)$")
ax.set_xlim([2, 10**5])
#ax.set_ylim([1, 2 * 10**1])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.plot(times, var[:, 1] / v**2 * np.sqrt(times) * (-np.log(v))**(1/6), c=colors[i], label=fr'$v={v}$', alpha=0.75)

ax.legend()
fig.savefig("Var.pdf")