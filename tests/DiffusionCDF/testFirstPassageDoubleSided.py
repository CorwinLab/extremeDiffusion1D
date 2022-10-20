from pyDiffusion import DiffusionTimeCDF
import numpy as np
import npquad

cdf = DiffusionTimeCDF('beta', [np.inf, np.inf], 20)
cdf.evolveAndSaveFirstPassageDoubleSided(10, [1, 2, 3, 4, 5, 6, 7, 10], 'Test.txt')