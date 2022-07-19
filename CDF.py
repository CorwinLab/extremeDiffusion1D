import sys 
sys.path.append("src")
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad
from matplotlib import pyplot as plt

cdf = DiffusionTimeCDF(1, 1000)
cdf.evolveToTime(1000)
vals = np.array(cdf.CDF)
times = np.arange(-1000, 1002, 2)

fontsize=18
fig, ax = plt.subplots()
ax.plot(times, vals)
ax.set_xlabel(r"$v(\theta)t$", fontsize=fontsize)
ax.set_ylabel(r"$P(v(\theta)t)$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=fontsize)
ax.set_ylim([0, 1.1])
ax.set_xlim([-1000, 1000])
fig.savefig("CDF.png", bbox_inches='tight')