import numpy as np
import npquad 
from libDiffusion import DiffusionTimeCDF

rec = DiffusionTimeCDF('beta', [1, 1], 10)
for _ in range(6):
    rec.iterateTimeStep()

print(rec.getCDF())
print(np.abs(np.diff(rec.getCDF())))
outside = rec.getProbOutsidePositions(6)
print(outside)