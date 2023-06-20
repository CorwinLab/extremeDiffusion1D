import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys 
sys.path.append("../../dataAnalysis")
from theory import log_moving_average, KPZ_var_fit

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

def KPZMean(s):
	return -s / 24 - 1/2 * np.log(2 * np.pi * s) 

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$-\mathbb{E}[(\ln(P_{\bf{B}}(X>vt^{3/4}, t))]$")
ax.set_xlim([10, 10**5])
ax.set_ylim([10**0/2, 2*10**2])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	#times, mean = log_moving_average(times, mean[:, 1], 10**(1/25))
	ax.plot(times, -mean[:, 1], c=colors[i], label=fr'$v={v}$', alpha=0.75)

xvals = np.array([10**4, 8*10**4])
ax.plot(xvals, xvals**(1/2) / 2, ls='--', c='k', label=r'$\sqrt{t}$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors + ['k'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

start_coord = (3*10**4, 2)
end_coord = (2*10**3, 7*10**1)
ax.annotate(
    "",
    xy=end_coord,
    xytext=start_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=40,
        headwidth=85,
        headlength=40,
        alpha=0.3,
    ),
    zorder=0,
)

fig.savefig("MeanScattering.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))$")
ax.set_xlim([2, 10**5])
ax.set_ylim([10**-4, 4 * 10**0])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.plot(times, var[:, 1], c=colors[i], label=fr'$v={v}$', alpha=0.75)
	
xvals = np.array([10**4, 8*10**4]).astype(float)
ax.plot(xvals, xvals**(-1/2) / 12.5, ls='--', c='k', label=r'$t^{-1/2}$')
leg = ax.legend(
    loc="upper right",
    framealpha=0,
    labelcolor=colors + ['k'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

start_coord = (5*10**2, 5 * 10**-4)
end_coord = (10**3, 2 * 10**-1)
ax.annotate(
    "",
    xy=end_coord,
    xytext=start_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=40,
        headwidth=85,
        headlength=40,
        alpha=0.3,
    ),
    zorder=0,
)

fig.savefig("VarScattering.svg", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$\frac{1}{v^2}\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t\rightarrow\infty))]$")
for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.scatter(np.log(1/v), var[-1, 1] / v**2, c='k')

#ax.legend()
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