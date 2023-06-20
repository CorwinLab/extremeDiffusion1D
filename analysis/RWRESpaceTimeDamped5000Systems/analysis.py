import os 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys 
sys.path.append("../../dataAnalysis")
from theory import log_moving_average

def gaussian(x, mean, var):
	return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x-mean)**2 / var)

gamma = '1'
dir = f'/home/jacob/Desktop/talapasMount/JacobData/RWRESpaceTimeDamped5000Systems/{gamma}/'
vs = os.listdir(dir)
vs = [i for i in vs if float(i) >= 0.2]
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
ax.set_ylabel(r"$\mathbb{E}[(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t))] + \frac{v^2\sqrt{t}}{2} + \frac{\ln(t)}{4} + \ln(v) + \frac{v^4}{12} + \frac{1}{2} \ln(2 \pi)$")
ax.set_xlim([10, 10**5])
ax.set_ylim([-1, 1])

for i, v in enumerate(vs): 
	mean = np.loadtxt(os.path.join(dir, v, 'Mean.txt'))
	times = mean[:, 0]
	v = float(v)
	times, mean = log_moving_average(times, -mean[:, 1], 10**(1/50))
	ax.plot(times, (-mean + v**2 / 2 * np.sqrt(times) + np.log(times)/4 + np.log(v) + v**4 / 12 + 1/2 * np.log(2 * np.pi)), c=colors[i], label=fr'$v={v}$', alpha=0.75)

ax.grid(True)
ax.legend()
fig.savefig(f"MeanRWREDamped.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t)))$")
ax.set_xlim([10, 10**5])
ax.set_ylim([3*10**-5, 1])

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	#times, var = log_moving_average(times, var[:, 1], 10**(1/25))
	ax.plot(times, var[:, 1], c=colors[i], label=fr'$v={v}$', alpha=0.75)

xvals = np.array([10**4, 7*10**4])
ax.plot(xvals, xvals ** (-1/2), ls='--', c='k', label=r'$t^{-1/2}$')

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

fig.savefig(f"VarRWREDamped.svg", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("v")
ax.set_ylabel(r"$\mathrm{Var}[\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t\rightarrow\infty))]$")

for i, v in enumerate(vs): 
	var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = var[:, 0]
	v = float(v)
	ax.scatter(v, var[-1, 1] / v**2 * np.sqrt(times[-1]), c='k')


ax.legend()
fig.savefig(f"VelocityVar{gamma}.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("t")
ax.set_ylabel("Prob")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(P_{\bf{B}}(X(t)>vt^{3/4}, t)))$")
#ax.set_ylim([2, 20])
ax.set_xlim([2, 10**5])

for i, v in enumerate(vs):
	if float(v) != 0.5:
		continue
	rwre_var = np.loadtxt(os.path.join(dir, v, 'Var.txt'))
	times = rwre_var[:, 0] 

	#ax.plot(times[1:], rwre_var[1:, 1], c=colors[i], label='RWRE Damped')

	var = np.loadtxt(os.path.join('/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocitiesQuads/', v, 'Var.txt'))
	ax.plot(var[1:, 0], var[1:, 1], c='k', label='Scattering Model')

	ax.plot(var[1:, 0], rwre_var[:, 1], c='r', label='Damped RWRE')

ax.legend()
fig.savefig(f"ScatteringComparison.svg", bbox_inches='tight')