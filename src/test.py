from libDiffusion import RandomDistribution
from matplotlib import pyplot as plt
import numpy as np

r = RandomDistribution("bates", [100])
vars = []
for _ in range(100000):
    vars.append(r.generateRandomVariable())

fig, ax = plt.subplots()
ax.hist(vars, bins=50)
fig.savefig("Test.png")