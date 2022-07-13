from matplotlib import pyplot as plt
import numpy as np 
import npquad
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

sys.path.append("../../src")

import theory

N = np.quad("1e24")
times = np.geomspace(1, np.log(N).astype(float)**3)
mean = theory.quantileMean(N, times)
var = theory.quantileVar(N, times)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Maximum Particle Position")

ax.plot(times, mean, c='k')
ax.fill_between(times, y1=mean+np.sqrt(var), y2 = mean-np.sqrt(var), alpha=0.7)
ax.set_xlim(min(times), max(times))
ax.set_ylim(min(mean), max(mean))

axins = zoomed_inset_axes(ax, zoom=17, loc='lower right')
axins.plot(times, mean, c='k')
axins.fill_between(times, y1=mean+np.sqrt(var), y2 = mean-np.sqrt(var), alpha=0.7)

plt.setp(axins.get_xticklabels(), visible=False)
plt.setp(axins.get_yticklabels(), visible=False)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
axins.set_xlim(10**3, 1.5*10**3)
axins.set_ylim(3.1*10**2, 4*10**2)

fig.savefig("FirstPassage.png")