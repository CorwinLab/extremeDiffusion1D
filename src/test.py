from itertools import count
from locale import normalize
from random import random
from libDiffusion import RandomDistribution
from matplotlib import pyplot as plt
import numpy as np

r = RandomDistribution("delta", [1/6, 2/3, 1/6])
vars = []
for _ in range(100000):
    vars.append(r.generateRandomVariable())

print(np.unique(vars, return_counts=True))
print(np.var(vars))
print(1/12)
fig, ax = plt.subplots()
ax.set_xlim([0, 1])
ax.hist(vars, bins=50)
fig.savefig("Test.png")