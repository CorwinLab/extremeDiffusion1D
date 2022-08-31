from random import random
from libDiffusion import RandomDistribution
from matplotlib import pyplot as plt
import numpy as np

b = 1/2 + 1/2*np.sqrt(5/63)
a = 1/2 - 1/2*np.sqrt(5/63)
r = RandomDistribution("quadratic", [a, b])
vars = []
for _ in range(1000000):
    vars.append(r.generateRandomVariable())

print(np.var(vars))
print(1/84)
fig, ax = plt.subplots()
ax.set_xlim([0, 1])
ax.hist(vars, bins=50)
fig.savefig("Test.png")