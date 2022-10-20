import matplotlib
matplotlib.use('Agg')
from libDiffusion import RandomDistribution
from matplotlib import pyplot as plt
import numpy as np

A = 5/12
r = RandomDistribution("delta", [A, 1-2*A, A])
vars = []
for _ in range(100000):
    vars.append(r.generateRandomVariable())

print(np.unique(vars, return_counts=True))
print(np.var(vars))
print(5/24)
fig, ax = plt.subplots()
ax.set_xlim([0, 1])
ax.hist(vars, bins=50)
fig.savefig("Test.png")