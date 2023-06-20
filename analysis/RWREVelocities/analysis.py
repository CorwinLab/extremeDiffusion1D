import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys 
sys.path.append("../../dataAnalysis")
from theory import log_moving_average, KPZ_var_fit

def gaussian(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x-mean)**2 / var)

dir = '/home/jacob/Desktop/talapasMount/JacobData/RWREVelocities'
vs = os.listdir(dir)
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
#ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathbb{E}[(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))] + \frac{v^2\sqrt{t}}{2} + \frac{\ln(t)}{4} - \ln(v) + \frac{v^4}{12} - h(0, v^4)$")
ax.set_xlim([10, 10**5])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	times, mean = log_moving_average(times, mean[:, 1], 10**(1/25))
	ax.plot(times, mean + v**2 / 2 * np.sqrt(times) + np.log(times)/4 - np.log(v) + v**4/12 - KPZMean(v**4), c=colors[i], label=fr'$v={v}$', alpha=0.75)

ax.grid(True)
ax.legend()
fig.savefig("RWREMean.pdf")

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t)))$")
ax.set_xlim([2, 10**5])

for i, v in enumerate(vs): 
	if v == '0.2':
		continue
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	theory_var = KPZ_var_fit(v**4)
	ax.hlines(theory_var, 10**4, 10**5, ls='--', color=colors[i])
	ax.plot(times, var[:, 1], c=colors[i], label=fr'$v={v}$', alpha=0.75)

leg = ax.legend(
    loc="upper right",
    framealpha=0,
    labelcolor=colors + ['k'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)
    
start_coord = (5*10**2, 5 * 10**-2)
end_coord = (10**3, 2)
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

fig.savefig("RWREVar.pdf")

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

'''
xvals = np.array([0.3, 0.6])
ax.plot(xvals, xvals**2, label=r'$v^2$')
'''
ax.legend()
fig.savefig("VelocityVar.pdf", bbox_inches='tight')