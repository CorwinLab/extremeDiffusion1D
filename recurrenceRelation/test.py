import pyrecurrence
import numpy as np
import npquad

r = pyrecurrence.Recurrance(beta=np.inf)
for _ in range(5):
    r.iterateTimeStep()
    q0 = r.findQuintile(5)
    q1 = r.findQuintile(10)
    q = r.findQuintiles([5, 10])
    print(q0, q1, q)
