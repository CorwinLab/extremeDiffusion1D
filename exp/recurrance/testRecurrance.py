import sys

sys.path.append("../../recuranceRelation")
sys.path.append("../../src")
from pyrecurrance import Recurrsion
import numpy as np
import npquad
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

rec = Recurrsion(1, 1_000)
rec.makeRec()
q = rec.findQuintiles([10, 100])

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Nth Quartile")
ax.plot(q[:, 1])
ax.plot(q[:, 0])
fig.savefig("test.png")
