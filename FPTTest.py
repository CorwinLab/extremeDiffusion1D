import numpy as np
from pyDiffusion import DiffusionPDF 

d = DiffusionPDF(1000, 'beta', [np.inf, np.inf], 20, bool(int(0)))
d.evolveAndSaveFirstPassage([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Time.txt')