import numpy as np
import npquad
import sys
sys.path.append("../src")
import pydiffusion as diff

d = diff.Diffusion(1000, 1, 1000)
d.iterateTimestep()
print(d.occupancy)
occ = np.array(d.occupancy)
print(occ.dtype)
