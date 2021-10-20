import sys
sys.path.append("./src")
from pydiffusionPDF import DiffusionPDF
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad

t = 10000
nParticles = np.quad("1e4500")
maxParticles = np.quad("1000")
d = DiffusionPDF(nParticles, np.inf, t, ProbDistFlag=True)
d.evolveToTime(t)
print("PDF variance", d.getGumbelVariance(maxParticles))

d = DiffusionTimeCDF(np.inf, t)
d.evolveToTime(t)
print("CDF Variance", d.getGumbelVariance(maxParticles))
