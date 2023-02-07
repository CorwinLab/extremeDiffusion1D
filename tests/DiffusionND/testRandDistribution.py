import numpy as np
import npquad

from pyDiffusion.pydiffusionND import RandDistribution

r = RandDistribution([1, 1, 1, 1])
print(r.getRandomNumbers())
print(sum(r.getRandomNumbers()))