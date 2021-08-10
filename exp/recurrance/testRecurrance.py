import sys

sys.path.append("../../recuranceRelation")
sys.path.append("../../src")
from pyrecurrance import Recurrsion
import numpy as np
import npquad
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

start = time.time()
tmax = 100_000 # Requires about 160GB of memory to work :(
beta = 1
rec = Recurrsion(beta, tmax)
rec.makeRec()
print('got here')
quintiles = [np.quad(f'1e{i}') for i in range(100, 4600, 100)]
q = rec.findQuintiles([10, 100])
print(time.time() - start)

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Nth Quartile")
ax.plot(q[:, 1])
ax.plot(q[:, 0])
fig.savefig("test.png")
