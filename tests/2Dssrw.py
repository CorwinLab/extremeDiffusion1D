import numpy as np
import npquad
from pyDiffusion.pydiffusionND import DiffusionND
import time
from simpleNDBC import twoDBCModelPDF

tMax = 500
size=5000
d = DiffusionND(4*[np.inf], size, 1000)

start = time.time()
for t in range(tMax):
    d.iterateTimestep()

run_time = time.time()-start
print(f"C++ Code: {run_time}s")

'''Eric's code'''
start = time.time()
new_occupancy = np.zeros((size, size))
new_occupancy[0, 0] = 1
twoDBCModelPDF(new_occupancy, tMax)
run_time = time.time() - start 
print(f"Eric's Code: {run_time}s")