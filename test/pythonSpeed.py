import time
import numpy as np

N = 100000
s = time.perf_counter()
a = np.random.beta(3, 2, size=N)
e = time.perf_counter()
print((e-s)/N * 1e9)
