from pyDiffusion import DiffusionTimeCDF
import numpy as np
import npquad

tMax = 10
cdf = DiffusionTimeCDF('beta', [np.inf, np.inf], tMax)
N = 10
for _ in range(tMax):
    cdf.iterateTimeStep()
    upper_pos = cdf.findQuantile(N)
    lower_pos = cdf.findLowerQuantile(N)
    print(cdf.CDF, lower_pos, upper_pos)
