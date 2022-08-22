from libDiffusion import RandomNumGenerator
import numpy as np

r = RandomNumGenerator(0.1)
vals = []
for _ in range(1000000):
    vals.append(r.generateBeta())
print(np.mean(vals))