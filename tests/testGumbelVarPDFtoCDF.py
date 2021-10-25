import sys
sys.path.append("../src")
sys.path.append("../DiffusionCDF")
from pydiffusionPDF import DiffusionPDF
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad
from matplotlib import pyplot as plt

t = 10
nParticles = np.quad("1e4500")
maxParticles = np.quad("10")

print("-------PDF Stuff------")
d = DiffusionPDF(nParticles, np.inf, t, ProbDistFlag=True)
d.evolveToTime(t)
var = d.getGumbelVariance(maxParticles)
cdf = d.getCDF()
print('PDF Variance:', var)

print("-----CDF Stuff--------")
d = DiffusionTimeCDF(np.inf, t)
d.evolveToTime(t)
print('CDF Variance:', d.getGumbelVariance(maxParticles))
percent_diff = (abs(np.array(d.CDF, dtype=np.quad)/np.array(cdf, dtype=np.quad)))
print('Min Percent difference', min(percent_diff))

fig, ax = plt.subplots()
bins = np.logspace(-300, 1)
ax.hist(percent_diff.astype(float), bins=bins)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("CDF / CDF from PDF")
ax.set_ylabel("Count")
ax.set_xlim([min(bins), max(bins)])
fig.savefig("hist.png")
