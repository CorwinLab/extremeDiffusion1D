import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def gaussian(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x-mean)**2 / var)

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocities'
vs = os.listdir(dir)
vs = [i for i in vs if float(i) > 0.2]
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
ax.set_ylabel(r"$-\mathbb{E}[(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))] - \frac{v^2\sqrt{t}}{2} - \frac{\ln(t)}{4} + \ln{v} - \frac{v^4}{12} - \frac{1}{2}\ln(2 \pi v^{4}) - t^{1/3}$")
ax.set_xlim([2, 2*10**4])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	if v <= 0.2:
		continue
	ax.scatter(times, (-mean[:, 1] - v**2 / 2 * np.sqrt(times) - np.log(times)/4 + np.log(v) - v**4 / 12 - np.log(2 * np.pi * v**4)/2 - times**(-1/3)), c=colors[i], label=fr'$v={v}$', alpha=0.75, s=1)

ax.legend()
fig.savefig("Mean.pdf")

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))] - \frac{v^2}{t}$")
ax.set_xlim([2, 2*10**4])
#ax.set_ylim([1, 2 * 10**1])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	if v <= 0.2:
		continue
	ax.plot(times, var[:, 1] - v**2/np.sqrt(times), c=colors[i], label=fr'$v={v}$', alpha=0.75)

xvals = np.array([10**2, 10**4]).astype(float)
ax.plot(xvals, xvals**(-1), ls='--', label=r'$t^{-1}$')
ax.plot(xvals, xvals**(-1/2)/2, ls='--', c='k', label=r'$\sqrt{t}$')
ax.legend()
fig.savefig("Var.pdf")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("v")
ax.set_ylabel(r"$\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t\rightarrow\infty))]$")
for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	var = var[var[:, 0] < 2*10**4]
	times = var[:, 0]
	v = float(v)
	if v <= 0.2:
		continue
	ax.scatter(v, var[-1, 1], c='k')

xvals = np.array([0.3, 0.6])
ax.plot(xvals, xvals**2 / 100, label=r'$v^2$')

ax.legend()
fig.savefig("VelocityVar.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylim([10**-1, 50])

ax.set_xlabel(r"$-\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t)) + \frac{v^2\sqrt{t}}{2} - \frac{\ln(t)}{4} - v^2 \sqrt{t}$")
for i, v in enumerate(vs): 
	probs = np.loadtxt(os.path.join(dir, v, 'Probs.txt'))
	t = np.loadtxt(os.path.join(dir, v, "T.txt"))

	v = float(v)
	if v <= 0.2:
		continue

	if np.all(probs == 0):
		continue 

	ax.hist((np.log(1-probs) - v**2 / 2 * np.sqrt(t) - np.log(t)/4), color=colors[i], density=True, label=fr'$v = {v}$', histtype='step', bins=50)
	
	xvals = np.linspace(-0.3, 0.7, 500)
	#ax.plot(xvals, gaussian(xvals, mean, var), c=colors[i], ls='--')

ax.legend()
fig.savefig("Probabilities.pdf", bbox_inches='tight')
