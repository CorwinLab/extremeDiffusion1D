import sys

sys.path.append("../src/")
from pydiffusionCDF import DiffusionTimeCDF
import numpy as np
import npquad
import time

beta = float("inf")
tMax = 13000000
d = DiffusionTimeCDF(beta, tMax)
d.id = "Test"
d.setTime(500000)
s = time.time()
d.saveState()
run_time = time.time() - s
s = time.time()
d.getGumbelVariance(np.quad("1e100"))
d.findQuantile(np.quad("1e100"))
run_timeQuantile = time.time() - s
np.savetxt("Time.txt", [run_time, run_timeQuantile])
