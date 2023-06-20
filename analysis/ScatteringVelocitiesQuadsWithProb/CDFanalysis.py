import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def gaussian(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x-mean)**2 / var)

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
ax.set_ylabel("Prob")
ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$-\mathbb{E}[(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))] - \frac{v^2\sqrt{t}}{2} - \frac{\ln(t)}{4}$")
ax.set_xlim([2, 10**5])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, (-mean[:, 1] - v**2 / 2 * np.sqrt(times) - np.log(times)/4), c=colors[i], label=fr'$v={v}$', alpha=0.75)

ax.legend()
fig.savefig("Mean.pdf")

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))]$")
ax.set_xlim([2, 10**5])
#ax.set_ylim([1, 2 * 10**1])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.plot(times, var[:, 1], c=colors[i], label=fr'$v={v}$', alpha=0.75)

xvals = np.array([10**3, 10**5]).astype(float)
ax.plot(xvals, xvals**(-1/2), ls='--', c='k', label=r'$\sqrt{t}$')
ax.legend()
fig.savefig("Var.pdf")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("v")
ax.set_ylabel(r"$\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t\rightarrow\infty))]$")
for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.scatter(v, var[-1, 1], c='k')

xvals = np.array([0.3, 0.6])
ax.plot(xvals, xvals**2 / 100, label=r'$v^2$')

ax.legend()
fig.savefig("VelocityVar.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathbb{E}[P_{\bf{B}}(X(t)>vt^{3/4}, t)]$")
ax.set_xlim([2, 10**5])
ax.set_ylim([10**-15, 10])
for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	ax.plot(times, np.exp(mean[:, 1]), c=colors[i], label=fr'$v={v}$', alpha=0.75)
	ax.plot(times, 2*gaussian(v*times**(3/4), 0, times), c=colors[i], ls='--')

ax.legend()
fig.savefig("PMean.pdf")